"""Pieces per work as a function of the merge threshold alpha (257 training works), for the digitisation figure.

For each work and each alpha, the work is projected onto the palette (majority filter radius 2) with pieces
smaller than alpha of the canvas merged; the number of connected single-colour pieces and the change of the
image (mean CIEDE2000 over foreground pixels between the image and its projection) are recorded.
Writes outputs/alpha_sweep.csv.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color

import priors

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def category(path):
    return "Pattern symbols" if "Pattern_symbols" in path else path.split("/")[-2]


def one(path):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    white = len(pal.lab)
    rgb = priors.load_rgb(ROOT / path, 512)
    lab = priors.to_lab(rgb)
    fg = priors.foreground_mask(lab)
    rows = []
    for a in ALPHAS:
        proj, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=a, smooth=2)
        n = sum(ndi.label(labels == k)[1] for k in np.unique(labels) if k != white)
        de = color.deltaE_ciede2000(lab[fg], priors.to_lab(proj)[fg]).mean() if fg.any() else np.nan
        rows.append({"path": path, "category": category(path), "alpha": a, "pieces": n, "change": float(de)})
    return rows


def main():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]]
    with Pool(16) as pool:
        rows = [r for rs in pool.map(one, paths) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/alpha_sweep.csv", index=False)
    print(df.groupby("alpha")[["pieces", "change"]].median().round(2))


if __name__ == "__main__":
    main()
