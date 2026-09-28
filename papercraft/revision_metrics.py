"""Extended per-image metrics for the revision (review #16-#20, #23).

python revision_metrics.py <method> [<method> ...]   ->  outputs/gen/<method>/rev_metrics.csv
python revision_metrics.py --embed <method> ...     ->  adds nn_train (DINOv2 max similarity to the
                                                     257 training works) and clip_style_dedup
python revision_metrics.py --projection-ablation     ->  outputs/rev_projection_ablation.csv
python revision_metrics.py --originals               ->  outputs/rev_originals_fragments.csv

Structure is always measured against the original, unperturbed drawing (outputs/contents).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
CDIR = ROOT / "outputs/contents"
ALPHA = 2e-4


def _pal():
    return priors.Palette.load(ROOT / "outputs/data/palette.json")


def drawing(cid):
    la = priors.load_rgb(CDIR / f"{cid}.png", 512)
    lines = la.mean(-1) < 0.85
    return lines, priors.silhouette_from_lineart(lines), priors.ground_from_lineart(lines)


def contour(mask):
    return mask & ~ndi.binary_erosion(mask)


def fscore(pred, gt, tol):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0
    dg = ndi.distance_transform_edt(~gt)
    dp = ndi.distance_transform_edt(~pred)
    p = float((dg[pred] <= tol).mean())
    r = float((dp[gt] <= tol).mean())
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def filled_fg(rgb):
    fg = priors.foreground_mask(priors.to_lab(rgb))
    return ndi.binary_fill_holes(morphology.binary_closing(fg, morphology.disk(2)))


def structure(rgb, lines, sil):
    edges = priors.edge_map(rgb)
    gt = morphology.skeletonize(lines)
    out = {}
    for tol in (2, 3, 5):
        p, r, f = fscore(edges, gt, tol)
        out[f"line_recall_t{tol}"], out[f"edge_prec_t{tol}"], out[f"edge_f1_t{tol}"] = r, p, f
    fg = filled_fg(rgb)
    _, _, out["contour_f"] = fscore(contour(fg), contour(sil), 3)
    inter = (fg & sil).sum()
    out["sil_iou"] = float(inter / max((fg | sil).sum(), 1))
    # decoration density: edge pixels inside the figure that are not near any drawing line, per 1000 figure pixels
    far = ndi.distance_transform_edt(~gt) > 3
    out["decoration_density"] = float(1000 * (edges & far & sil).sum() / max(sil.sum(), 1))
    return out


def pieces_below(labels, alpha, white):
    n_small = 0
    for k in np.unique(labels):
        if k == white:
            continue
        cc, n = ndi.label(labels == k)
        if n:
            a = np.bincount(cc.ravel())[1:]
            n_small += int((a < alpha * labels.size).sum())
    return n_small


def per_image(args):
    path, cid = args
    pal = _pal()
    white = len(pal.lab_with_white) - 1
    lines, sil, ground = drawing(cid)
    rgb = priors.load_rgb(path, 512)
    lab = priors.to_lab(rgb)
    fg = priors.foreground_mask(lab)
    _, de = priors.nearest_palette(lab, pal.lab_with_white)
    rec = {"file": Path(path).name, "content": cid}
    rec.update({f"raw_{k}": v for k, v in structure(rgb, lines, sil).items()})
    rec["raw_palette_conformance"] = float((de[fg] < 5).mean()) if fg.any() else np.nan
    proj, labels, _, _ = priors.cut_project(rgb, pal, ground=ground)
    rec.update({f"proj_{k}": v for k, v in structure(proj, lines, sil).items()})
    lab_p = priors.to_lab(proj)
    fg_p = priors.foreground_mask(lab_p)
    rec["proj_purity"] = priors.colour_purity(lab_p, fg_p) if fg_p.sum() > 50 else np.nan
    rec["proj_colours_used"] = int(sum((labels == k).mean() > 1e-3 for k in np.unique(labels) if k != white))
    rec["proj_pieces_below_alpha"] = pieces_below(labels, ALPHA, white)
    rec["proj_cut_residual_fg"] = float(priors.color.deltaE_ciede2000(lab, lab_p)[fg].mean()) if fg.any() else np.nan
    rec["proj_cut_residual_all"] = float(priors.color.deltaE_ciede2000(lab, lab_p).mean())
    proj2, labels2, _, _ = priors.cut_project(proj, pal, ground=ground)
    rec["proj_idempotence_changed"] = float((labels2 != labels).mean())
    return rec


def run_methods(methods):
    for m in methods:
        files = sorted((G / m).glob("*_seed*.png"))
        args = [(str(f), f.name.split("_s")[0] if "_s_seed" in f.name else f.name.rsplit("_s", 1)[0]) for f in files]
        args = [(p, cid.rsplit("_s", 1)[0] if not (CDIR / f"{cid}.png").exists() else cid) for p, cid in args]
        with ProcessPoolExecutor(24) as ex:
            rows = list(ex.map(per_image, args, chunksize=2))
        df = pd.DataFrame(rows)
        base = pd.read_csv(G / m / "metrics.csv")[["file", "style", "seed", "set"]] if (G / m / "metrics.csv").exists() else None
        if base is not None:
            df = df.merge(base, on="file", how="left")
        df.to_csv(G / m / "rev_metrics.csv", index=False)
        print(m, len(df), "line_recall raw/proj %.3f/%.3f" % (df.raw_line_recall_t3.mean(), df.proj_line_recall_t3.mean()),
              "pieces<alpha %.2f" % df.proj_pieces_below_alpha.mean(), "idem %.5f" % df.proj_idempotence_changed.mean())


def embed(methods):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    refs = json.loads((ROOT / "outputs/audit/heldout_refs.json").read_text())
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def ims(paths, project=None):
        out = []
        for p in paths:
            rgb = priors.load_rgb(p, 512)
            out.append(Image.fromarray((rgb * 255).astype(np.uint8)))
        return out

    @torch.no_grad()
    def d_emb(images):
        return torch.cat([torch.nn.functional.normalize(dino(**dproc(images=images[i:i + 32], return_tensors="pt")).pooler_output, dim=-1)
                          for i in range(0, len(images), 32)])

    @torch.no_grad()
    def c_emb(images):
        return torch.cat([torch.nn.functional.normalize(clip.get_image_features(**cproc(images=images[i:i + 32], return_tensors="pt")), dim=-1)
                          for i in range(0, len(images), 32)])

    cache = ROOT / "outputs/audit/embed_cache.pt"
    if cache.exists():
        T, C = torch.load(cache)
    else:
        T = d_emb(ims([ROOT / it["path"] for it in split["train"]]))
        C = torch.nn.functional.normalize(c_emb(ims([ROOT / p for p in refs])).mean(0), dim=0)
        torch.save((T, C), cache)
    for m in methods:
        f = G / m / "rev_metrics.csv"
        df = pd.read_csv(f)
        images = ims([G / m / x for x in df.file])
        D, Cl = d_emb(images), c_emb(images)
        sims = (D @ T.T)
        df["nn_train_dino"] = sims.max(1).values.numpy()
        df["nn_train_idx"] = sims.argmax(1).numpy()
        df["clip_style_dedup"] = (Cl @ C).numpy()
        df.to_csv(f, index=False)
        print(m, "nn_train %.3f (max %.3f)" % (df.nn_train_dino.mean(), df.nn_train_dino.max()),
              "clip_style_dedup %.3f" % df.clip_style_dedup.mean())


def _abl(args):
    path, cid = args
    pal = _pal()
    lines, sil, ground = drawing(cid)
    rgb = priors.load_rgb(path, 512)
    lab = priors.to_lab(rgb)
    labels, _ = priors.nearest_palette(lab, pal.lab_with_white)
    to_rgb = lambda l: np.clip(priors.color.lab2rgb(pal.lab_with_white[l][None])[0], 0, 1).astype(np.float32)
    variants = {
        "raw": rgb,
        "colour only": to_rgb(labels),
        "ground only": np.where(ground[..., None], 1.0, rgb).astype(np.float32),
        "colour + merge (no ground)": priors.cut_project(rgb, pal)[0],
        "full projection": priors.cut_project(rgb, pal, ground=ground)[0],
    }
    rows = []
    for k, x in variants.items():
        s = structure(x, lines, sil)
        rows.append({"file": Path(path).name, "variant": k, **s})
    return rows


def projection_ablation():
    out = []
    for m in ("cutcraft", "cutline"):
        files = sorted((G / m).glob("*_seed*.png"))
        args = [(str(f), f.name.split("_s_seed")[0]) for f in files]
        with ProcessPoolExecutor(24) as ex:
            for rs in ex.map(_abl, args, chunksize=2):
                for r in rs:
                    r["method"] = m
                    out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "outputs/rev_projection_ablation.csv", index=False)
    print(df.groupby(["method", "variant"], sort=False)[["line_recall_t3", "edge_prec_t3", "edge_f1_t3", "contour_f", "sil_iou"]].mean().round(3).to_string())


def _frag(args):
    p, size = args
    pal = _pal()
    rgb = priors.load_rgb(p, size)
    lab = priors.to_lab(rgb)
    raw, _ = priors.nearest_palette(lab, pal.lab_with_white)
    mode = priors._mode_filter(raw, len(pal.lab_with_white), 2)
    white = len(pal.lab_with_white) - 1
    thin, total = 0, 0
    for k in np.unique(mode):
        if k == white:
            continue
        cc, n = ndi.label(mode == k)
        if not n:
            continue
        areas = np.bincount(cc.ravel())[1:]
        small = np.nonzero(areas < ALPHA * mode.size)[0] + 1
        total += len(small)
        if len(small):
            m = np.isin(cc, small)
            # a sliver is a small piece that disappears under a 1-px erosion (at most 2 px wide)
            er = ndi.binary_erosion(m)
            lab_er, _ = ndi.label(er)
            survivors = set(np.unique(cc[er])) - {0}
            thin += len(set(small.tolist()) - survivors)
    return {"file": str(p), "size": size, "fragments": total, "slivers_le2px": thin}


def originals():
    fs = sorted((G / "ksl_original").glob("*.png"))
    args = [(str(f), s) for f in fs for s in (512, 1024)]
    with ProcessPoolExecutor(24) as ex:
        df = pd.DataFrame(list(ex.map(_frag, args)))
    df.to_csv(ROOT / "outputs/rev_originals_fragments.csv", index=False)
    g = df.groupby("size")[["fragments", "slivers_le2px"]].mean()
    g["sliver_share"] = g.slivers_le2px / g.fragments
    print(g.round(3).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--embed":
        embed(a[1:])
    elif a and a[0] == "--projection-ablation":
        projection_ablation()
    elif a and a[0] == "--originals":
        originals()
    else:
        run_methods(a)
