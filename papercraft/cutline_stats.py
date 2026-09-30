"""How cut-line maps differ from generic edges, measured on the 257 training works.

For each work, the dense and coarse cut-line maps (outputs/cutlines, 1024 px, 3-px strokes) are compared
with Canny edges of the same image (cv2.Canny 100/200 on grey, the edges a canny ControlNet is trained on):
  canny_spurious   share of Canny edge pixels farther than 3 px from any dense cut line
                   (edges inside a single paper piece: gloss, creases, shading, JPEG noise)
  cut_missed       share of dense cut-line pixels farther than 3 px from any Canny edge
                   (piece boundaries that Canny misses, e.g. between colours of similar lightness)
  dense_len_mm, coarse_len_mm, canny_len_mm   total line length at a 200 x 200 mm print
  dense_pieces, coarse_pieces                 pieces of the projected label map
Writes outputs/cutline_stats.csv and prints medians per category.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
CL = ROOT / "outputs/cutlines"
MM = 200.0 / 1024


def category(path):
    return "Pattern symbols" if "Pattern_symbols" in path else path.split("/")[-2]


def lines(p):
    return np.asarray(Image.open(p).convert("L")) < 128


def one(path):
    stem = path.replace("/", "__")
    d, c, k = (lines(CL / f"{stem}__{n}.png") for n in ("dense", "coarse", "canny"))
    ds, cs, ks = (morphology.skeletonize(x) for x in (d, c, k))
    near_d = ndi.distance_transform_edt(~ds) <= 3
    near_k = ndi.distance_transform_edt(~ks) <= 3
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    rgb = priors.load_rgb(ROOT / path, 512)
    pieces = {}
    for name, frac, sm in (("dense", 2e-4, 2), ("coarse", 5e-3, 3)):
        _, lab, _, _ = priors.cut_project(rgb, pal, min_area_frac=frac, smooth=sm)
        n = 0
        for v in np.unique(lab):
            if v == len(pal.lab):
                continue
            n += ndi.label(lab == v)[1]
        pieces[name] = n
    return {"path": path, "category": category(path),
            "canny_spurious": float((ks & ~near_d).sum() / max(ks.sum(), 1)),
            "cut_missed": float((ds & ~near_k).sum() / max(ds.sum(), 1)),
            "dense_len_mm": float(ds.sum() * MM), "coarse_len_mm": float(cs.sum() * MM), "canny_len_mm": float(ks.sum() * MM),
            "dense_pieces": pieces["dense"], "coarse_pieces": pieces["coarse"]}


def main():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]]
    with Pool(16) as pool:
        df = pd.DataFrame(pool.map(one, paths))
    df.to_csv(ROOT / "outputs/cutline_stats.csv", index=False)
    cols = ["canny_spurious", "cut_missed", "dense_len_mm", "coarse_len_mm", "canny_len_mm", "dense_pieces", "coarse_pieces"]
    with pd.option_context("display.width", 200):
        t = df.groupby("category")[cols].median().round(2); t["n"] = df.groupby("category").size()
        print(t)
        print("ALL median", df[cols].median().round(3).to_dict())
        print("ALL IQR", df[cols].quantile([.25, .75]).round(3).to_dict())


if __name__ == "__main__":
    main()
