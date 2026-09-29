"""Colour atlas of the digital corpus (in the spirit of Zou et al. 2025, Fig. 4): the paper-stock palette
with HSV decomposition and corpus share, and the palette proportions of every catalogue category as rings.
Shares come from outputs/data/corpus_palette_shares.npy (order: split train + held_out, 282 images).
"""
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#1f1f1e", "#6b6a64"
CATS = [("person", "Figures"), ("animals", "Animals"), ("plants", "Plants"), ("daily", "Daily objects"),
        ("window decoration", "Window flowers"), ("frame", "Borders"), ("Pattern symbols", "Pattern symbols")]


def category(path):
    if "Pattern_symbols" in path:
        return "Pattern symbols"
    return path.split("/")[-2]


def main():
    pal = json.loads((ROOT / "outputs/data/palette.json").read_text())
    hexes = pal["hex"]
    H = np.load(ROOT / "outputs/data/corpus_palette_shares.npy")[:, :len(hexes)]
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]] + split["held_out"]
    assert len(paths) == len(H)
    cats = np.array([category(p) for p in paths])
    rgb = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in hexes]
    hsv = [colorsys.rgb_to_hsv(*c) for c in rgb]
    overall = H.mean(0) / H.mean(0).sum()
    order = sorted(range(len(hexes)), key=lambda k: (hsv[k][1] < 0.15, hsv[k][0], hsv[k][2]))  # chromatic by hue, then neutrals

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.03, 0.4, 0.94, 0.53])
    for x, k in enumerate(order):
        ax.add_patch(plt.Rectangle((x, 0.35), 0.9, 0.62, color=rgb[k], ec="#c9c8c0", lw=0.4))
        h, s, v = hsv[k]
        ax.text(x + 0.45, 0.25, hexes[k].upper(), ha="center", va="top", fontsize=5.2, color=INK, family="monospace")
        ax.text(x + 0.45, 0.1, f"H {h*360:.0f}°\nS {s*100:.0f}  V {v*100:.0f}", ha="center", va="top", fontsize=4.6,
                color=MUTED, linespacing=1.3)
        ax.bar(x + 0.45, overall[k] * 3.2, width=0.9, bottom=1.03, color=rgb[k], ec="#c9c8c0", lw=0.4)
        ax.text(x + 0.45, 1.05 + overall[k] * 3.2, f"{overall[k]*100:.1f}%", ha="center", va="bottom", fontsize=5.2, color=INK)
    ax.set_xlim(-0.1, len(order)); ax.set_ylim(-0.3, 1.03 + overall.max() * 3.2 + 0.18); ax.axis("off")
    ax.set_title("a  Paper-stock palette of the digital corpus: share of foreground pixels (bars), hex and HSV",
                 loc="left", fontsize=8, color=INK)
    fig.text(0.03, 0.335, "b  Palette proportions by catalogue category (n = images)", fontsize=8, color=INK)
    for i, (key, lab) in enumerate(CATS):
        sel = cats == key
        share = H[sel].mean(0); share = share / share.sum()
        ax2 = fig.add_axes([0.02 + i * 0.139, 0.075, 0.13, 0.23])
        ax2.pie([share[k] for k in order], colors=[rgb[k] for k in order], startangle=90, counterclock=False,
                wedgeprops=dict(width=0.38, edgecolor="white", linewidth=0.6))
        ax2.text(0, -1.32, f"{lab}\nn = {sel.sum()}", ha="center", va="top", fontsize=6, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper/figures/fig_atlas.{ext}", dpi=240)
    print({lab: int((cats == k).sum()) for k, lab in CATS}, "total", len(cats))


if __name__ == "__main__":
    main()
