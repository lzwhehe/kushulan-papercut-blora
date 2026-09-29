"""From line drawing to cut paper (design K01 of the cutting kit: teapot, CutCraft-SDXL, seed 0).

a drawing; b generated design; c enforced 200 mm cutting plan (re-rasterised SVG); d-f the three paper colours
with most pieces, each piece filled in its colour and outlined, as cut from separate sheets; g numbered assembly
view of the printable kit; h summary of the plan.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

import priors
from svg_validate import PX_MM, rasterise

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#1f1f1e", "#6b6a64"


def page_crop(pdf, i, box_mm=(5, 40, 205, 240)):
    pg = pymupdf.open(pdf)[i]
    pix = pg.get_pixmap(dpi=220, clip=pymupdf.Rect(*(v / 25.4 * 72 for v in box_mm)))
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    lab = pal.lab_with_white
    rgb_pal = np.clip(priors.color.lab2rgb(lab[None])[0], 0, 1)
    white = len(lab) - 1
    svg = ROOT / "outputs/svg_enforced/cutcraft/ood_teapot_s_seed0.svg"
    ras = rasterise(svg).astype(np.float32) / 255
    labels, _ = priors.nearest_palette(priors.to_lab(ras), lab)
    sheets = {}
    for k in np.unique(labels):
        if k != white:
            sheets[int(k)] = ndi.label(labels == k)[1]
    top = sorted(sheets, key=lambda k: -sheets[k])[:3]
    manifest = {m["code"]: m for m in json.loads((ROOT / "outputs/cutting_kit/manifest_UNBLIND.json").read_text())}
    k01 = manifest["K01"]

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    ax = axes.ravel()
    ax[0].imshow(Image.open(ROOT / "outputs/contents/ood_teapot.png").convert("RGB")); ax[0].set_title("a  Line drawing")
    ax[1].imshow(Image.open(ROOT / "outputs/gen/cutcraft/ood_teapot_s_seed0.png").convert("RGB")); ax[1].set_title("b  Generated design")
    ax[2].imshow(ras); ax[2].set_title("c  Cutting plan, 200 mm")
    ax[3].imshow(page_crop(ROOT / "outputs/cutting_kit/K01.pdf", 0, box_mm=(80, 95, 140, 155))); ax[3].set_title("d  Numbered assembly view (detail)")
    for j, k in enumerate(top):
        m = labels == k
        canvas = np.ones(m.shape + (3,))
        canvas[m] = rgb_pal[k]
        a = ax[4 + j]
        a.imshow(canvas)
        for c in measure.find_contours(np.pad(m, 1).astype(float), 0.5):
            a.plot(c[:, 1] - 1, c[:, 0] - 1, color=INK, lw=0.3)
        a.set_title(f"{'efg'[j]}  Sheet {j + 1}: {sheets[k]} pieces")
    ax[7].axis("off")
    ax[7].text(0.02, 0.8, "h  Plan summary", fontsize=6.5, color=INK, transform=ax[7].transAxes)
    ax[7].text(0.02, 0.2, f"{k01['pieces']} pieces on {k01['sheets']} paper colours\n"
                          f"smallest piece {k01['smallest_mm2']:.1f} mm$^2$ (limit 8 mm$^2$)\n"
                          f"minimum width 1 mm enforced\n"
                          f"1 px = {PX_MM:.3f} mm; SVG in millimetres\n"
                          f"not yet cut (candidate design)", fontsize=5.8, color=MUTED,
               transform=ax[7].transAxes, linespacing=1.6)
    for a in ax[:7]:
        a.axis("off"); a.title.set_fontsize(6.5); a.title.set_color(INK); a.title.set_ha("left"); a.title.set_position((0, 1))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.06, hspace=0.18)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper/figures/fig_fabrication.{ext}", dpi=300)
    print("saved", {k: sheets[k] for k in top})


if __name__ == "__main__":
    main()
