"""Automatic metrics and expert-rating agreement for content/style/output triplets.

Pairs CSV (e.g. written by tools/generate.py) needs the columns
`id,content,style,output` and optionally `prompt`; paths are relative to the CSV.

    python tools/evaluate.py metrics runs/xyz/pairs.csv --out runs/xyz/metrics.csv
    python tools/evaluate.py metrics pairs.csv --clip          # + CLIP-I / CLIP-T (needs open_clip + torch)
    python tools/evaluate.py metrics pairs.csv --dino          # + DINOv2 content score (needs transformers + torch)
    python tools/evaluate.py agreement ratings.csv             # Kendall's W per dimension

Structure  Edge-F1 between output edges and content line art (tolerance in px),
           silhouette IoU between the filled outer shapes, optional DINOv2 cosine.
           Edge-F1 / IoU assume the output is spatially aligned with the line art.
           Plain B-LoRA sampling is not spatially conditioned, so these two are only
           meaningful for aligned generation (ControlNet / img2img). Every run
           therefore also scores a mismatched-content control (each output against
           another sample's line art); a structural metric is only informative if
           matched > control. The summary reports both and their gap.
Style      CLIP-I(output, style) and a 4x4x4 RGB palette Hellinger distance.
Text       CLIP-T(output, prompt).
Ratings CSV columns: `sample,rater,dimension,score` (one row per rating).

Requires numpy and Pillow. Every metric is computed at a fixed resolution
(--size) so that numbers are comparable across runs; report that value.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image


# ----------------------------------------------------------------- image utils
def load_rgb(path, size):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def gray(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def dilate(mask, r):
    """Binary dilation with a (2r+1)^2 square, numpy only."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    p = np.pad(mask, r)
    h, w = mask.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def sobel_edges(g, quantile=0.9):
    """Edge map: top (1-quantile) of Sobel gradient magnitude, ignoring flat images."""
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.hypot(gx, gy)
    if mag.max() < 1e-6:
        return np.zeros_like(g, dtype=bool)
    return mag >= max(np.quantile(mag, quantile), 0.25)


def line_mask(rgb, thresh=0.6):
    """Strokes of a line drawing: dark pixels."""
    return gray(rgb) < thresh


def fill_outside(barrier):
    """Pixels reachable from the image border without crossing `barrier` (geodesic flood fill)."""
    free = ~barrier
    seed = np.zeros_like(free)
    seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = free[0, :], free[-1, :], free[:, 0], free[:, -1]
    reach = seed
    while True:
        grown = dilate(reach, 1) & free
        if (grown == reach).all():
            return reach
        reach = grown


def silhouette(rgb, is_line_art, tol=0.12):
    """Filled outer shape. Line art: close small gaps then fill. Colour image: differ from border colour."""
    if is_line_art:
        closed = ~fill_outside(dilate(line_mask(rgb), 2))
        return ~dilate(~closed, 2)  # undo the outward growth of the gap-closing dilation
    else:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0)
        barrier = np.abs(rgb - bg).max(axis=2) > tol
    return ~fill_outside(barrier)


# ---------------------------------------------------------------------- metrics
def edge_f1(output_rgb, content_rgb, tol=3):
    pred = sobel_edges(gray(output_rgb))
    ref = line_mask(content_rgb)
    if not ref.any() or not pred.any():
        return {"edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0}
    precision = float((pred & dilate(ref, tol)).sum() / pred.sum())
    recall = float((ref & dilate(pred, tol)).sum() / ref.sum())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


def silhouette_iou(output_rgb, content_rgb):
    a, b = silhouette(output_rgb, False), silhouette(content_rgb, True)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def palette_distance(a_rgb, b_rgb, bins=4, ignore_white=True):
    """Hellinger distance (0 = same palette, 1 = disjoint) between coarse RGB histograms."""
    def hist(x):
        px = x.reshape(-1, 3)
        if ignore_white:
            keep = px.min(axis=1) < 0.92
            px = px[keep] if keep.any() else px
        q = np.minimum((px * bins).astype(int), bins - 1)
        h = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(float)
        return h / h.sum()
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(hist(a_rgb) * hist(b_rgb)).sum())))


class Clip:
    def __init__(self, model="ViT-L-14", pretrained="openai", device=None):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tok = open_clip.get_tokenizer(model)
        self.name = f"{model}/{pretrained}"

    def image(self, path):
        with self.torch.no_grad(), Image.open(path) as im:
            x = self.pre(im.convert("RGB")).unsqueeze(0).to(self.device)
            f = self.model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    def text(self, s):
        with self.torch.no_grad():
            f = self.model.encode_text(self.tok([s]).to(self.device))
        return f / f.norm(dim=-1, keepdim=True)

    @staticmethod
    def cos(a, b):
        return float((a * b).sum())


class Dino:
    def __init__(self, model="facebook/dinov2-base", device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval().to(self.device)
        self.name = model

    def image(self, path):
        with self.torch.no_grad(), Image.open(path) as im:
            x = self.proc(images=im.convert("RGB"), return_tensors="pt").to(self.device)
            f = self.model(**x).last_hidden_state[:, 0]
        return f / f.norm(dim=-1, keepdim=True)


def read_pairs(pairs_csv):
    base = pairs_csv.parent
    with pairs_csv.open(encoding="utf-8-sig", newline="") as f:
        return [{**row, **{k: base / row[k] for k in ("content", "style", "output")}} for row in csv.DictReader(f)]


def structure_metrics(o, c, tol):
    return {**edge_f1(o, c, tol), "silhouette_iou": silhouette_iou(o, c)}


def evaluate_pairs(pairs_csv, size=512, tol=3, clip=None, dino=None, control=True):
    """Per-sample metrics; with `control`, also against a cyclic shift of the content images."""
    pairs = read_pairs(pairs_csv)
    imgs = [{k: load_rgb(p[k], size) for k in ("content", "style", "output")} for p in pairs]
    feats = {}
    rows = []
    for i, (p, im) in enumerate(zip(pairs, imgs)):
        m = {"id": p["id"], **structure_metrics(im["output"], im["content"], tol),
             "palette_dist": palette_distance(im["output"], im["style"])}
        if dino:
            for k in ("content", "output"):
                feats.setdefault((k, i), dino.image(p[k]))
            m["dino_content"] = Clip.cos(feats[("output", i)], feats[("content", i)])
        if clip:
            fo = clip.image(p["output"])
            m["clip_i"] = Clip.cos(fo, clip.image(p["style"]))
            if p.get("prompt"):
                m["clip_t"] = Clip.cos(fo, clip.text(p["prompt"]))
        if control and len(pairs) > 1:
            j = (i + 1) % len(pairs)
            for k, v in structure_metrics(im["output"], imgs[j]["content"], tol).items():
                m[f"control_{k}"] = v
            if dino:
                feats.setdefault(("content", j), dino.image(pairs[j]["content"]))
                m["control_dino_content"] = Clip.cos(feats[("output", i)], feats[("content", j)])
        rows.append(m)
    return rows


def bootstrap_ci(values, n=2000, seed=0, alpha=0.05):
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def summarize(rows):
    keys = [k for k in rows[0] if k != "id"] if rows else []
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and not math.isnan(r[k])]
        lo, hi = bootstrap_ci(vals)
        out[k] = {"mean": sum(vals) / len(vals), "ci95": [lo, hi], "n": len(vals)}
    for k in keys:
        if f"control_{k}" in out:
            gaps = [r[k] - r[f"control_{k}"] for r in rows if f"control_{k}" in r]
            lo, hi = bootstrap_ci(gaps)
            out[k]["gap_vs_control"] = {"mean": sum(gaps) / len(gaps), "ci95": [lo, hi],
                                        "informative": lo > 0}
    return out


# -------------------------------------------------------------- rater agreement
def kendalls_w(matrix):
    """Kendall's W with tie correction. matrix[rater][item] = score; all raters rate all items."""
    m, n = len(matrix), len(matrix[0])
    if m < 2 or n < 2:
        raise ValueError("need at least 2 raters and 2 items")
    ranks, ties = [], 0.0
    for scores in matrix:
        order = sorted(range(n), key=lambda i: scores[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            t = j - i + 1
            ties += t ** 3 - t
            i = j + 1
        ranks.append(r)
    totals = [sum(r[i] for r in ranks) for i in range(n)]
    mean = sum(totals) / n
    s = sum((t - mean) ** 2 for t in totals)
    denom = m * m * (n ** 3 - n) - m * ties
    return 12 * s / denom if denom else 1.0


def agreement(ratings_csv):
    data = defaultdict(lambda: defaultdict(dict))
    with ratings_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            data[row["dimension"]][row["rater"]][row["sample"]] = float(row["score"])
    out = {}
    for dim, by_rater in data.items():
        items = sorted(set.intersection(*(set(v) for v in by_rater.values())))
        raters = sorted(by_rater)
        matrix = [[by_rater[r][i] for i in items] for r in raters]
        scores = [x for row in matrix for x in row]
        out[dim] = {"kendalls_w": kendalls_w(matrix), "mean": sum(scores) / len(scores),
                    "raters": len(raters), "items": len(items)}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("metrics")
    p.add_argument("pairs", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--tol", type=int, default=3)
    p.add_argument("--clip", action="store_true")
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--dino", action="store_true")
    p.add_argument("--dino-model", default="facebook/dinov2-base")
    p.add_argument("--no-control", action="store_true")
    a = sub.add_parser("agreement")
    a.add_argument("ratings", type=Path)
    args = parser.parse_args()
    if args.cmd == "metrics":
        clip = Clip(args.clip_model, args.clip_pretrained) if args.clip else None
        dino = Dino(args.dino_model) if args.dino else None
        rows = evaluate_pairs(args.pairs, args.size, args.tol, clip, dino, not args.no_control)
        out = args.out or args.pairs.with_name("metrics.csv")
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
        summary = {"pairs": str(args.pairs), "size": args.size, "edge_tol_px": args.tol,
                   "clip": clip.name if clip else None, "dino": dino.name if dino else None,
                   "metrics": summarize(rows)}
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(agreement(args.ratings), indent=2, ensure_ascii=False))
