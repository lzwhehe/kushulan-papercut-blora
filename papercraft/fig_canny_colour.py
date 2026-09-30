"""Cut lines vs generic edges as training pairs: effect on colour (plan v5, Fig. 9).

a  Examples: drawing, Canny-edge LoRA, cut-line LoRA (same model, settings and seed).
b  Per-drawing distributions (34 drawings; originals: 19 in-domain works) of paper colours per design,
   colour purity and palette distance, for the originals, the untrained model, the Canny LoRA and the
   cut-line LoRA. Writes paper/figures/fig_canny_colour.pdf/.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_tables_v5 import drawing_table

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
EX = ["ood_rooster", "ood_tiger", "ood_crane_bird"]  # rooster (used throughout), a median and the largest difference in paper colours (seed 0)
GROUPS = [("originals", "ksl_original", "#8c8c8c"), ("untrained", "qwen_q0", "#b9b3a6"),
          ("Canny LoRA", "qwen_q1_canny", "#7b89a2"), ("cut-line LoRA", "qwen_q1", "#201963")]
METRICS = [("lay_colours", "paper colours per design", False), ("purity", "colour purity", True), ("palette_de", "palette dist. ΔE$_{00}$", False)]


def main():
    fig = plt.figure(figsize=(7.2, 3.6))
    ga = fig.add_gridspec(len(EX), 3, left=0.03, right=0.43, top=0.9, bottom=0.03, wspace=0.04, hspace=0.05)
    for i, cid in enumerate(EX):
        for j, (lab, p) in enumerate([("drawing", ROOT / f"outputs/contents/{cid}.png"),
                                      ("Canny LoRA", G / "qwen_q1_canny" / f"{cid}_s_seed0.png"),
                                      ("cut-line LoRA", G / "qwen_q1" / f"{cid}_s_seed0.png")]):
            ax = fig.add_subplot(ga[i, j])
            ax.imshow(Image.open(p).convert("RGB").resize((320, 320)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3); sp.set_color("#bbbbbb")
            if i == 0:
                ax.set_title(lab, fontsize=7)
    tabs = {m: drawing_table(m) for _, m, _ in GROUPS}
    gb = fig.add_gridspec(1, 3, left=0.51, right=0.99, top=0.84, bottom=0.2, wspace=0.55)
    for k, (col, lab, _) in enumerate(METRICS):
        ax = fig.add_subplot(gb[0, k])
        data = [tabs[m][col].dropna().values for _, m, _ in GROUPS]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.2))
        for patch, (_, _, c) in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(c); patch.set_edgecolor(c)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color("#6b6a64"); w.set_linewidth(0.7)
        for x, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), x) + np.random.default_rng(x).uniform(-0.18, 0.18, len(d)), d, s=3, color="#1f1f1e", alpha=0.35, zorder=3)
        ax.set_xticks(range(1, len(GROUPS) + 1)); ax.set_xticklabels([g[0] for g in GROUPS], rotation=40, ha="right", fontsize=6)
        ax.set_title(lab, fontsize=7); ax.tick_params(axis="y", labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.955, "a  same model and seed, different training lines", fontsize=7, weight="bold")
    fig.text(0.51, 0.955, "b  colour of the outputs (per drawing)", fontsize=7, weight="bold")
    fig.savefig(ROOT / "paper/figures/fig_canny_colour.pdf")
    fig.savefig(ROOT / "paper/figures/fig_canny_colour.png", dpi=220)
    print("saved")


if __name__ == "__main__":
    main()
