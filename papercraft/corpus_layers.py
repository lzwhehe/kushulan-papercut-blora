"""Craft statistics of the catalogue itself under the layered projection.

Every catalogue entry (282 images) is projected with cut_project (nearest paper colour, figure on
white board, speckle removal, pieces < 8 mm^2 at 200 mm merged) and analysed with layered.analyse.
This gives the corpus's own distribution of pieces, layers, base pieces and colours per work, which
is the reference against which generated plans are compared. Writes outputs/layered/corpus.csv.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import layered
import priors

ROOT = Path(__file__).resolve().parents[1]


def category(path):
    return "Pattern_symbols" if "Pattern_symbols" in path else path.split("/")[-2]


def one(path):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    rgb = priors.load_rgb(ROOT / path, 512)
    fg = priors.foreground_mask(priors.to_lab(rgb))
    fg = ndi.binary_closing(fg, iterations=3)
    fg = ndi.binary_fill_holes(fg)
    ground = ~ndi.binary_dilation(fg, iterations=12)
    _, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=2e-4, smooth=2, ground=ground)
    white = len(pal.lab_with_white) - 1
    r = layered.analyse(labels.astype(np.uint8), white)
    if r is None:
        return None
    stats, _ = r
    stats["path"] = path
    stats["category"] = category(path)
    return stats


def main():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]] + split["held_out"]
    with Pool(12) as pool:
        rows = [r for r in pool.map(one, paths) if r is not None]
    df = pd.DataFrame(rows)
    (ROOT / "outputs/layered").mkdir(exist_ok=True)
    df.to_csv(ROOT / "outputs/layered/corpus.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.groupby("category")[["pieces", "layers", "base_pieces", "decoration_share", "colours", "base_colours", "cut_length_flat_mm", "cut_length_layered_mm"]].median().round(2))
        print("ALL", df[["pieces", "layers", "base_pieces", "decoration_share", "colours", "base_colours"]].median().round(2).to_dict())
    print("CORPUSDONE")


if __name__ == "__main__":
    main()
