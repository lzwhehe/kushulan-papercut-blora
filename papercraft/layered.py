"""Layered (paste-up) projection: from a flat colour mosaic to a stack of pasted pieces.

Ku Shulan's colour paste-up is made by cutting pieces and pasting them in layers on a white board:
a base shape first, then smaller pieces on top (套色贴), repeated pieces side by side (排比贴) and
dots as the last decoration (点缀贴) [docs/craft_process_sources.md]. The flat label map produced by
cut_project() records only what is visible; this module recovers a plausible paste order from it.

Rule (containment tree). Every connected single-colour region is a piece. Its parent is the piece
that surrounds it: pieces are assigned breadth-first from the white board; among the unassigned
pieces the one whose boundary is most enclosed by already-assigned pieces is taken next, and its
parent is the assigned neighbour sharing the longest boundary, unless at least GROUND_SHARE of its
boundary touches the board, in which case it is pasted directly on the board (a base piece).
Depth 1 = base pieces, depth 2 = pieces pasted on a base piece, and so on. The physical shape of a
piece is its region together with everything pasted on top of it (holes are filled), which is what
the scissors actually cut.

Outputs per label map: pieces, layers (max depth), base pieces, pieces per depth, colours, cut
length (sum of piece perimeters, mm at 200 mm) for the flat and the layered plan, piece areas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

import priors

ROOT = Path(__file__).resolve().parents[1]
GROUND_SHARE = 0.25
SIZE_MM = 200.0


def regions(labels, white):
    """Connected single-colour regions. Returns (region id map, list of colour per region); 0 = board."""
    rid = np.zeros(labels.shape, np.int32)
    colours = [white]
    nxt = 1
    for k in np.unique(labels):
        if k == white:
            continue
        cc, n = ndi.label(labels == k)
        rid[cc > 0] = cc[cc > 0] + nxt - 1
        colours += [int(k)] * n
        nxt += n
    return rid, np.array(colours)


def boundary_shares(rid):
    """shares[i] = {j: number of boundary pixel-pairs between region i and region j} (4-neighbourhood)."""
    n = rid.max() + 1
    pairs = []
    for a, b in ((rid[:, :-1], rid[:, 1:]), (rid[:-1, :], rid[1:, :])):
        m = a != b
        pairs.append(np.stack([a[m], b[m]], 1))
    p = np.concatenate(pairs)
    p = np.concatenate([p, p[:, ::-1]])
    key = p[:, 0].astype(np.int64) * n + p[:, 1]
    u, c = np.unique(key, return_counts=True)
    shares = [dict() for _ in range(n)]
    for k, cnt in zip(u, c):
        i, j = divmod(int(k), n)
        shares[i][j] = int(cnt)
    return shares


def containment_tree(labels, white):
    rid, colours = regions(labels, white)
    n = len(colours)
    shares = boundary_shares(rid)
    total = np.array([sum(s.values()) for s in shares], float)
    parent = np.full(n, -1)
    depth = np.zeros(n, int)
    parent[0] = 0
    assigned = np.zeros(n, bool)
    assigned[0] = True
    # a region with no boundary (whole image one colour) or isolated: attach to board
    remaining = set(range(1, n))
    while remaining:
        best, best_frac = None, -1.0
        for i in remaining:
            if total[i] == 0:
                best, best_frac = i, 2.0
                break
            f = sum(c for j, c in shares[i].items() if assigned[j]) / total[i]
            if f > best_frac:
                best, best_frac = i, f
        i = best
        s = shares[i]
        if total[i] == 0 or best_frac == 0:
            parent[i] = 0
        else:
            g = s.get(0, 0) / total[i]
            if g >= GROUND_SHARE:
                parent[i] = 0
            else:
                cand = [(c, j) for j, c in s.items() if assigned[j]]
                parent[i] = max(cand)[1]
        depth[i] = depth[parent[i]] + 1
        assigned[i] = True
        remaining.discard(i)
    return rid, colours, parent, depth


def piece_shapes(rid, parent, depth):
    """Boolean mask per piece including all descendants (the shape the scissors cut)."""
    n = len(parent)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    masks = {}

    def fill(i):
        m = rid == i
        for c in children[i]:
            m |= fill(c)
        masks[i] = m
        return m

    sys.setrecursionlimit(max(10000, n + 100))
    for i in children[0]:
        fill(i)
    return masks


def perimeter_px(mask):
    return sum(len(c) for c in measure.find_contours(np.pad(mask, 1).astype(float), 0.5))


def analyse(labels, white, mm_per_px=None):
    mm_per_px = mm_per_px or SIZE_MM / labels.shape[0]
    rid, colours, parent, depth = containment_tree(labels, white)
    n = len(colours) - 1
    if n == 0:
        return None
    masks = piece_shapes(rid, parent, depth)
    areas_flat = np.bincount(rid.ravel(), minlength=len(colours))[1:] * mm_per_px ** 2
    areas_layer = np.array([masks[i].sum() for i in range(1, len(colours))]) * mm_per_px ** 2
    per_flat = np.array([perimeter_px(rid == i) for i in range(1, len(colours))]) * mm_per_px
    per_layer = np.array([perimeter_px(masks[i]) for i in range(1, len(colours))]) * mm_per_px
    d = depth[1:]
    counts = np.bincount(d, minlength=6)[1:6]
    return {
        "pieces": int(n), "layers": int(d.max()), "base_pieces": int((d == 1).sum()),
        "depth2": int(counts[1]), "depth3": int(counts[2]), "depth4plus": int(counts[3:].sum()),
        "decoration_share": float((d >= 2).mean()),
        "colours": int(len(np.unique(colours[1:]))),
        "base_colours": int(len(np.unique(colours[1:][d == 1]))),
        "base_area_share": float(areas_flat[d == 1].sum() / areas_flat.sum()),
        "largest_piece_mm2": float(areas_layer.max()), "median_piece_mm2": float(np.median(areas_layer)),
        "cut_length_flat_mm": float(per_flat.sum()), "cut_length_layered_mm": float(per_layer.sum()),
        "holes_flat": int(sum(max(0, len(measure.find_contours(np.pad(rid == i, 1).astype(float), 0.5)) - 1) for i in range(1, len(colours)))),
    }, (rid, colours, parent, depth, masks)


def _one(args):
    f, white = args
    labels = np.load(f)
    r = analyse(labels, white)
    if r is None:
        return None
    stats, _ = r
    name = Path(f).name
    stats["file"] = name.replace(".labels.npy", ".png")
    stats["content"] = name.split("_s_seed")[0] if "_s_seed" in name else name.rsplit("_seed", 1)[0]
    return stats


def run_method(method, out_dir, limit=None, workers=12):
    from multiprocessing import Pool
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    white = len(pal.lab_with_white) - 1
    files = sorted((ROOT / "outputs/gen" / method).glob("*.labels.npy"))[:limit]
    with Pool(workers) as pool:
        rows = [r for r in pool.map(_one, [(str(f), white) for f in files]) if r is not None]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{method}.csv", index=False)
    return df


if __name__ == "__main__":
    methods = sys.argv[1:] or ["ksl_original", "cutcraft", "cutline", "qwen_q1", "qwen_q0", "qwen_q1_canny", "cl_canny", "blora", "instantstyle"]
    out = ROOT / "outputs/layered"
    summary = []
    for m in methods:
        df = run_method(m, out)
        s = df.drop(columns=["file", "content"]).median(numeric_only=True)
        s["method"] = m; s["n"] = len(df)
        summary.append(s)
        print(f"{m:16s} n={len(df):3d} pieces {s.pieces:5.0f} layers {s.layers:3.0f} base {s.base_pieces:4.0f} deco {s.decoration_share:.2f} "
              f"colours {s.colours:3.0f} base_col {s.base_colours:2.0f} cut flat {s.cut_length_flat_mm:6.0f} layered {s.cut_length_layered_mm:6.0f} holes {s.holes_flat:4.0f}", flush=True)
    pd.DataFrame(summary).to_csv(out / "summary_median.csv", index=False)
