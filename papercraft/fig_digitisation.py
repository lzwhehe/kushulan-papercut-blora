"""Technical figure: digitising the craft (paper-stock palette and cut lines), step by step on one training work.

a  Foreground pixels of the 257 training works in the CIELAB a*-b* plane (density) with the 16 k-means centres
   (marker area = corpus share) and the palette strip.
b  Nearest paper colour: work, label map and CIEDE2000 distance to the assigned colour, with a zoomed crop.
c  Choice of the merge threshold: pieces per work and projection change against alpha (257 works, median and
   interquartile range); the fine and coarse levels are marked.
d  Training pairs: fine, coarse and flipped cut lines paired with the work (4 pairs per work, 1,028 in total).
The cleaning steps and the fine and coarse cut lines themselves are shown in fig_cutline_process.py (Fig. 4).
Writes paper/figures/fig_digitisation.pdf/.png.
"""
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, segmentation

import cutlines
import priors

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements/animals/2.jpg"
INK, MUTED = "#1f1f1e", "#6b6a64"


def palette_codes(pal):
    rgb = pal.rgb
    hsv = [colorsys.rgb_to_hsv(*c) for c in rgb]
    order = sorted(range(len(rgb)), key=lambda k: (hsv[k][1] < 0.15, hsv[k][0], hsv[k][2]))
    return order, {k: f"P{i + 1}" for i, k in enumerate(order)}


def sample_pixels(n_per=250, seed=0):
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    rng = np.random.default_rng(seed)
    out = []
    for it in split["train"]:
        lab = priors.to_lab(priors.load_rgb(ROOT / it["path"], 256))
        fg = priors.foreground_mask(lab)
        px = lab[fg]
        if len(px):
            out.append(px[rng.choice(len(px), min(n_per, len(px)), replace=False)])
    return np.concatenate(out)


def best_crop(changed, size=112):
    """Top-left corner of the size x size window with the most merged pixels."""
    s = ndi.uniform_filter(changed.astype(float), size)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    h, w = changed.shape
    return int(np.clip(y - size // 2, 0, h - size)), int(np.clip(x - size // 2, 0, w - size))


def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color("#bbbbbb")


def main():
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans"})
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    order, code = palette_codes(pal)
    pal_lab = pal.lab_with_white
    pal_rgb = np.clip(color.lab2rgb(pal_lab[None])[0], 0, 1)
    white = len(pal_lab) - 1
    rgb = priors.load_rgb(WORK, 512)
    lab = priors.to_lab(rgb)
    raw, de = priors.nearest_palette(lab, pal_lab)
    filt = priors._mode_filter(raw, len(pal_lab), 2)
    _, final, _, _ = priors.cut_project(rgb, pal, min_area_frac=2e-4, smooth=2)
    changed = filt != final
    y0, x0 = best_crop(changed)
    sl = (slice(y0, y0 + 112), slice(x0, x0 + 112))
    maps = cutlines.cutline_maps(WORK, pal, res=512)

    H, SHIFT = 5.6, 1.8  # inches; the old 7.4-in canvas without its middle rows
    top = lambda y: (y * 7.4 - SHIFT) / H  # rows a, b
    bot = lambda y: y * 7.4 / H            # rows c, d
    hs = lambda h: h * 7.4 / H             # heights
    fig = plt.figure(figsize=(7.2, H))
    # ---- a palette
    axa = fig.add_axes([0.07, top(0.635), 0.29, hs(0.295)])
    px = sample_pixels()
    axa.hexbin(px[:, 1], px[:, 2], gridsize=70, bins="log", cmap="Greys", mincnt=1, linewidths=0)
    for k in order:
        axa.scatter(pal.lab[k, 1], pal.lab[k, 2], s=25 + 1400 * pal.weight[k], color=pal.rgb[k], edgecolor=INK, lw=0.6, zorder=3)
        axa.annotate(code[k], (pal.lab[k, 1], pal.lab[k, 2]), xytext=(3, 3), textcoords="offset points", fontsize=5.6, zorder=4)
    axa.set_xlabel("a*"); axa.set_ylabel("b*")
    axa.set_title("a  k-means (K = 16) on foreground pixels\n    of the 257 training works (CIELAB)", loc="left", fontsize=7)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axs = fig.add_axes([0.07, top(0.545), 0.29, hs(0.02)])
    for i, k in enumerate(order):
        axs.add_patch(Rectangle((i, 0), 0.94, 1, color=pal.rgb[k]))
        axs.text(i + 0.47, -0.4, code[k][1:], ha="center", va="top", fontsize=5.6)  # 1-16, "P" once at the left
    axs.text(-0.3, -0.4, "P", ha="right", va="top", fontsize=5.6)
    axs.set_xlim(0, 16); axs.set_ylim(0, 1); axs.axis("off")
    fig.text(0.07, top(0.505), "foreground: not ($L^*>90$ and $C^*<10$); ≤ 4,000 pixels per work", fontsize=5.6, color=MUTED)

    # ---- b assignment
    titles_b = ["work", "nearest paper colour $k(u)$", "$\\Delta E_{00}$ to $k(u)$ (0–20)"]
    imgs_b = [rgb, pal_rgb[raw], plt.cm.viridis(np.clip(de / 20, 0, 1))[..., :3]]
    for j, (t, im) in enumerate(zip(titles_b, imgs_b)):
        ax = fig.add_axes([0.42 + j * 0.19, top(0.745), 0.175, hs(0.175)]); ax.imshow(im); clean_ax(ax)
        ax.set_title(("b  " if j == 0 else "") + t, loc="left", fontsize=6.5)
        ax.add_patch(Rectangle((x0, y0), 112, 112, fill=False, ec="#c52929", lw=0.8))
        axz = fig.add_axes([0.42 + j * 0.19, top(0.565), 0.175, hs(0.175)]); axz.imshow(im[sl], interpolation="nearest"); clean_ax(axz)
        for sp in axz.spines.values():
            sp.set_color("#c52929"); sp.set_linewidth(0.8)
    fig.text(0.42, top(0.545), "zoom: every pixel is assigned to the nearest of 16 paper colours or the white board (CIEDE2000)", fontsize=5.6, color=MUTED)

    # ---- c alpha sweep
    sw = pd.read_csv(ROOT / "outputs/alpha_sweep.csv")
    g = sw.groupby("alpha")
    for j, (col, lab_y) in enumerate((("pieces", "pieces per work"), ("change", "projection change $\\Delta E_{00}$"))):
        ax = fig.add_axes([0.08 + j * 0.2, bot(0.065), 0.14, hs(0.13)])
        med, q1, q3 = g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)
        ax.fill_between(med.index, q1, q3, color="#7b89a2", alpha=0.35, lw=0)
        ax.plot(med.index, med.values, "o-", color="#201963", ms=2.5, lw=1)
        ax.set_xscale("log"); ax.set_xlabel("α (share of canvas)", fontsize=6); ax.set_ylabel(lab_y, fontsize=6)
        for a, c in ((2e-4, "#201963"), (5e-3, "#c52929")):
            ax.axvline(a, color=c, lw=0.7, ls="--")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=5.5)
        if j == 0:
            ax.set_title("c  choice of α (257 works; median, IQR)", loc="left", fontsize=6.5, x=-0.3, y=1.12)
    fig.text(0.08, bot(0.005), "dashed: fine (blue) and coarse (red) levels", fontsize=5.6, color=MUTED)

    # ---- d pairs
    work_img = Image.fromarray((rgb * 255).astype(np.uint8))
    fine = maps["dense"].convert("RGB"); coarse = maps["coarse"].convert("RGB")
    cells = [(fine, "fine"), (coarse, "coarse"), (fine.transpose(Image.FLIP_LEFT_RIGHT), "flipped*")]
    for j, (im, t) in enumerate(cells):
        ax = fig.add_axes([0.50 + j * 0.105, bot(0.075), 0.095, hs(0.095)]); ax.imshow(im); clean_ax(ax); ax.set_title(t, fontsize=5.5)
    ax = fig.add_axes([0.85, bot(0.075), 0.095, hs(0.095)]); ax.imshow(work_img); clean_ax(ax); ax.set_title("work", fontsize=5.5)
    fig.text(0.815, bot(0.12), "→", fontsize=12, ha="center", va="center")
    fig.text(0.50, bot(0.205), "d  training pairs: input = cut lines, target = work", fontsize=6.5, va="bottom")
    fig.text(0.50, bot(0.03), "2 levels × 2 flips = 4 pairs per work; 257 works → 1,028 pairs (*work flipped too)", fontsize=5.6, color=MUTED)
    fig.savefig(ROOT / "paper/figures/fig_digitisation.pdf")
    fig.savefig(ROOT / "paper/figures/fig_digitisation.png", dpi=220)
    print("saved; crop", y0, x0)


if __name__ == "__main__":
    main()
