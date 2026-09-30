"""Figure: flat mosaic vs layered paste-up for originals and generated designs.

Columns: image, flat pieces (each piece outlined), depth map (base = 1 ... top), exploded layers
(depth 1, 2, 3+ shown separately, each piece drawn with its filled shape in its paper colour).
Rows: three originals (in-domain test works) and the CutCraft-SDXL and CutCraft-Qwen designs of the
same drawings. Writes paper/figures/fig_layers.jpg.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb
from skimage import segmentation

import layered
import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
DEPTH_COLS = ["#ffffff", "#201963", "#c52929", "#e6af25", "#4d7b35", "#7b89a2", "#7b89a2", "#7b89a2"]


def layer_image(rid, colours, depth, masks, pal_rgb, which):
    """RGB image of the pieces whose depth is in `which`, drawn as filled shapes on white."""
    out = np.ones(rid.shape + (3,), np.float32)
    order = sorted([i for i in range(1, len(colours)) if depth[i] in which], key=lambda i: depth[i])
    for i in order:
        out[masks[i]] = pal_rgb[colours[i]]
    return out


def row(labels, white, pal_rgb):
    stats, (rid, colours, parent, depth, masks) = layered.analyse(labels, white)
    flat = pal_rgb[np.where(labels == white, len(pal_rgb) - 1, labels)]
    flat = segmentation.mark_boundaries(flat, rid, color=(0, 0, 0), mode="inner")
    dm = np.zeros(rid.shape, int)
    for i in range(1, len(colours)):
        dm[rid == i] = min(depth[i], 5)
    depth_rgb = np.array([to_rgb(c) for c in DEPTH_COLS])[dm]
    l1 = layer_image(rid, colours, depth, masks, pal_rgb, {1})
    l2 = layer_image(rid, colours, depth, masks, pal_rgb, {2})
    l3 = layer_image(rid, colours, depth, masks, pal_rgb, set(range(3, 20)))
    return stats, [flat, depth_rgb, l1, l2, l3]


def main(ids=("ind_人物-16", "ind_动物-38", "ind_窗花-3")):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    white = len(pal.lab_with_white) - 1
    pal_rgb = np.concatenate([pal.rgb, [[1, 1, 1]]], 0)
    rows, labels_txt = [], []
    for cid in ids:
        for m, lab in (("ksl_original", "original"), ("cutcraft", "CutCraft-SDXL"), ("qwen_q1", "CutCraft-Qwen")):
            f = G / m / f"{cid}_s_seed0.labels.npy"
            if not f.exists():
                f = G / m / f"{cid}_seed0.labels.npy"
            img = Image.open(str(f).replace(".labels.npy", ".png")).convert("RGB").resize((512, 512))
            stats, ims = row(np.load(f), white, pal_rgb)
            rows.append([np.asarray(img) / 255.0] + ims)
            labels_txt.append(f"{lab}\n{cid.split('_', 1)[1]}\n{stats['pieces']} pieces, {stats['layers']} layers\n{stats['base_pieces']} base")
    cols = ["image", "flat pieces", "paste depth", "layer 1 (base)", "layer 2", "layers 3+"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.55 * len(cols), 1.55 * len(rows)))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=7)
        axes[i, 0].set_ylabel(labels_txt[i], fontsize=5.5)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.05)
    fig.savefig(ROOT / "paper/figures/fig_layers.jpg", dpi=200, pil_kwargs={"quality": 90})
    print("saved")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("ind_人物-16", "ind_动物-38", "ind_窗花-3"))
