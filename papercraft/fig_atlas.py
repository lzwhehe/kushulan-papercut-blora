"""Colour atlas of the digital corpus (redesigned; draws on conventions used across heritage colour studies:
CIELAB chromaticity plots, colour co-occurrence networks and per-category proportion bars).

a  the 16 paper-stock colours in the CIELAB a*-b* plane, marker area proportional to corpus share, labelled P1-P16
   (ordered by hue, neutrals last); a companion strip gives L* and C*.
b  co-occurrence network: two colours are linked when they both cover >= 3 % of a work; edge width = share of works
   in which they co-occur, node area = corpus share.
c  100 % stacked bars of palette proportions per catalogue category, same colour order.
Shares come from outputs/data/corpus_palette_shares.npy (order: split train + held_out, 282 images).
Also writes paper/sections/table_palette.tex (P-code, swatch, hex, L*a*b*, share).
"""
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
CATS = [("person", "Figures"), ("animals", "Animals"), ("plants", "Plants"), ("daily", "Daily objects"),
        ("window decoration", "Window flowers"), ("frame", "Borders"), ("Pattern_symbols", "Pattern symbols")]
CO_MIN = 0.03


LABEL_DIR = {"P1": (-1, 1), "P8": (-1.25, 0)}


def category(path):
    return "Pattern_symbols" if "Pattern_symbols" in path else path.split("/")[-2]


def main():
    pal = json.loads((ROOT / "outputs/data/palette.json").read_text())
    hexes, lab = pal["hex"], np.array(pal["lab"])
    K = len(hexes)
    H = np.load(ROOT / "outputs/data/corpus_palette_shares.npy")[:, :K]
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]] + split["held_out"]
    cats = np.array([category(p) for p in paths])
    rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes])
    hsv = [colorsys.rgb_to_hsv(*c) for c in rgb]
    order = sorted(range(K), key=lambda k: (hsv[k][1] < 0.15, hsv[k][0], hsv[k][2]))
    code = {k: f"P{i + 1}" for i, k in enumerate(order)}
    Hn = H / H.sum(1, keepdims=True)
    share = Hn.mean(0)
    present = Hn >= CO_MIN
    co = (present.T.astype(float) @ present.astype(float)) / len(Hn)  # share of works where both present

    fig = plt.figure(figsize=(7.2, 6.0))
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    # ---- a: a*-b* plane
    ax = fig.add_axes([0.07, 0.58, 0.40, 0.38])
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    for k in order:
        a, b = lab[k, 1], lab[k, 2]
        s = 90 + 3200 * share[k]
        ax.scatter(a, b, s=s, color=rgb[k], edgecolor=INK, lw=0.5, zorder=3, clip_on=False)
        off = 0.71 * np.sqrt(s) / 2 + 1.5  # label just outside the marker (radius in points)
        dx, dy = LABEL_DIR.get(code[k], (1, 1))  # crowded markers get their label on another side
        ax.annotate(code[k], (a, b), xytext=(dx * off, dy * off), textcoords="offset points", fontsize=6, color=INK,
                    ha="right" if dx < 0 else "left", va="center" if dy == 0 else "baseline", zorder=5)
    ax.margins(x=0.10, y=0.12)  # room for the largest markers at the edges
    ax.set_xlabel("a* (green – red)", fontsize=7); ax.set_ylabel("b* (blue – yellow)", fontsize=7)
    ax.set_title("a  Paper-stock colours in CIELAB (area = corpus share)", loc="left", fontsize=7.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6, color=MUTED)
    # L*/C* strip
    ax2 = fig.add_axes([0.07, 0.415, 0.40, 0.065])
    for i, k in enumerate(order):
        ax2.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=rgb[k]))
        ax2.text(i + 0.46, 1.12, code[k], ha="center", fontsize=5.6, color=INK)
        ax2.text(i + 0.46, -0.38, f"{lab[k,0]:.0f}", ha="center", fontsize=5.6, color=MUTED)
        ax2.text(i + 0.46, -0.86, f"{np.hypot(lab[k,1], lab[k,2]):.0f}", ha="center", fontsize=5.6, color=MUTED)
    ax2.text(-0.25, -0.38, "$L^*$", ha="right", fontsize=5.6, color=MUTED); ax2.text(-0.25, -0.86, "$C^*$", ha="right", fontsize=5.6, color=MUTED)
    ax2.set_xlim(0, K); ax2.set_ylim(-1.0, 1.4); ax2.axis("off")
    # ---- b: co-occurrence network (circular layout in hue order)
    axn = fig.add_axes([0.55, 0.43, 0.42, 0.5])
    ang = {k: 2 * np.pi * i / K for i, k in enumerate(order)}
    pos = {k: (np.cos(ang[k]), np.sin(ang[k])) for k in order}
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K) if co[i, j] >= 0.15]
    for i, j in sorted(pairs, key=lambda p: co[p]):
        axn.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=INK, alpha=0.12 + 0.6 * (co[i, j] - 0.15) / (co.max() - 0.15),
                 lw=0.4 + 3.5 * (co[i, j] - 0.15) / (co.max() - 0.15), zorder=1, solid_capstyle="round")
    for k in order:
        s = 60 + 2600 * share[k]
        axn.scatter(*pos[k], s=s, color=rgb[k], edgecolor=INK, lw=0.5, zorder=3)
        # label just outside the marker: radius in points over points per data unit (axes 0.42 x 7.2 in for 2.7 units)
        r = 1 + (np.sqrt(s) / 2) / (0.42 * 7.2 * 72 / 2.7) + 0.13
        axn.text(pos[k][0] * r, pos[k][1] * r, code[k], ha="center", va="center", fontsize=6, color=INK, zorder=5)
    axn.set_xlim(-1.35, 1.35); axn.set_ylim(-1.35, 1.35); axn.set_aspect("equal"); axn.axis("off")
    axn.set_title("b  Colours used together in one work\n    (link width: share of works with both, ≥ 15 %)", loc="left", fontsize=7.5, color=INK)
    # ---- c: stacked bars per category
    axc = fig.add_axes([0.2, 0.06, 0.77, 0.29])
    ys = np.arange(len(CATS))[::-1]
    for y, (key, lab_) in zip(ys, CATS):
        sel = cats == key
        s = Hn[sel].mean(0)
        left = 0
        for k in order:
            axc.barh(y, s[k], left=left, color=rgb[k], edgecolor="white", lw=0.5, height=0.7)
            if s[k] >= 0.08:
                axc.text(left + s[k] / 2, y, f"{s[k]*100:.0f}", ha="center", va="center", fontsize=5.6,
                         color="white" if lab[k, 0] < 55 else INK)
            left += s[k]
        axc.text(-0.01, y, f"{lab_} (n = {sel.sum()})", ha="right", va="center", fontsize=6.5, color=INK)
    axc.set_xlim(0, 1); axc.set_ylim(-0.6, len(CATS) - 0.4); axc.set_yticks([])
    axc.set_xticks([0, 0.25, 0.5, 0.75, 1]); axc.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=6, color=MUTED)
    for s in ("top", "right", "left"):
        axc.spines[s].set_visible(False)
    axc.set_title("c  Palette proportions by catalogue category (share of foreground pixels, %)", loc="left", fontsize=7.5, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper/figures/fig_atlas.{ext}", dpi=240)
    # palette table for the supplement
    lines = [r"\definecolor{pp%d}{HTML}{%s}" % (k, hexes[k].lstrip("#").upper()) for k in range(K)]
    rows = [f"{code[k]} & \\textcolor{{pp{k}}}{{\\rule{{4mm}}{{2.4mm}}}} & {hexes[k].upper().replace('#', chr(92) + '#')} & {lab[k,0]:.0f} & {lab[k,1]:.0f} & {lab[k,2]:.0f} & {share[k]*100:.1f}\\\\" for k in order]
    (ROOT / "paper/sections/table_palette_colors.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "paper/sections/table_palette.tex").write_text("\n".join(rows) + "\n")
    strong = sorted(pairs, key=lambda p: -co[p])[:5]
    print("top pairs:", [(code[i], code[j], round(co[i, j], 2)) for i, j in strong], "max", round(co.max(), 2), "n links", len(pairs))


if __name__ == "__main__":
    main()
