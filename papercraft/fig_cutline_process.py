"""Figure: how cut lines are derived from a catalogue work, and how they differ from generic edges.

Rows: training works from four categories. Columns:
  a work; b nearest paper colour (raw label map); c after speckle removal and merging of pieces
  below 8 mm^2 (dense projection); d dense cut lines; e coarse cut lines (pieces below 0.5 % of the
  canvas merged); f Canny edges of the same image; g difference: black = shared, red = Canny only
  (edges inside one paper piece), blue = cut line only (boundaries Canny misses).
Writes paper/figures/fig_cutline_process.jpg.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
CL = ROOT / "outputs/cutlines"
BASE = "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
WORKS = [("person/24.jpg", "Figure"), ("plants/9.jpg", "Plant")]  # two works only, to limit reproduction of the artist's works


def lines(path, res=512):
    im = Image.open(path).convert("L").resize((res, res), Image.NEAREST)
    return np.asarray(im) < 128


def diff_image(cut, canny):
    cs, ks = morphology.skeletonize(cut), morphology.skeletonize(canny)
    near_c = ndi.distance_transform_edt(~cs) <= 2
    near_k = ndi.distance_transform_edt(~ks) <= 2
    out = np.ones(cut.shape + (3,))
    shared = (cs & near_k) | (ks & near_c)
    only_k = ks & ~near_c
    only_c = cs & ~near_k
    grow = lambda m: ndi.binary_dilation(m, iterations=1)
    out[grow(shared)] = (0.15, 0.15, 0.15)
    out[grow(only_k)] = (0.84, 0.16, 0.16)
    out[grow(only_c)] = (0.16, 0.35, 0.84)
    return out


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    pal_rgb = np.concatenate([pal.rgb, [[1, 1, 1]]], 0)
    rows = []
    for rel, lab in WORKS:
        path = f"{BASE}/{rel}"
        rgb = priors.load_rgb(ROOT / path, 512)
        _, dense_lab, raw, _ = priors.cut_project(rgb, pal, min_area_frac=2e-4, smooth=2)
        stem = path.replace("/", "__")
        d, c, k = (lines(CL / f"{stem}__{n}.png") for n in ("dense", "coarse", "canny"))
        rows.append((lab, [rgb, pal_rgb[raw], pal_rgb[dense_lab], ~d, ~c, ~k, diff_image(d, k)]))
    cols = ["a  work", "b  nearest paper colour", "c  filtered, pieces\n    < 8 mm$^2$ merged", "d  dense cut lines",
            "e  coarse cut lines", "f  generic edges (Canny)", "g  d vs f"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.3 * len(cols), 1.36 * len(rows)))
    for i, (lab, ims) in enumerate(rows):
        for j, im in enumerate(ims):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3); sp.set_color("#bbbbbb")
            if i == 0:
                ax.set_title(cols[j], fontsize=6.3, loc="left")
        axes[i, 0].set_ylabel(lab, fontsize=7)
    fig.text(0.995, 0.004, "g: black = both  |  red = Canny only (details below the cuttable size, texture)  |  blue = cut line only (colours of similar lightness)",
             ha="right", va="bottom", fontsize=5.8, color="#444444")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.93, bottom=0.03, wspace=0.04, hspace=0.05)
    fig.savefig(ROOT / "paper/figures/fig_cutline_process.jpg", dpi=250, pil_kwargs={"quality": 92})
    print("saved")


if __name__ == "__main__":
    main()
