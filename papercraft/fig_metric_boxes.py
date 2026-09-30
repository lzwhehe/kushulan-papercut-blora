"""Per-drawing distributions of the four primary measures for every method (box plots), plan v5.

Row a: Experiment 1 (19 in-domain drawings); row b: Experiment 2 (12 new-subject + 3 repository drawings).
Dashed line: median of the reference redrawings (reference standard). Methods: B-LoRA, StyleAligned, InstantStyle, SDXL LoRA + ControlNet, the untrained editing model and ours,
with the reference redrawings (19 in-domain works) as the reference standard. Writes
paper/figures/fig_metric_boxes.pdf/.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_tables3 import per_drawing

ROOT = Path(__file__).resolve().parents[1]
METHODS = [("references", "ksl_original", "#8c8c8c"), ("B-LoRA", "blora", "#c9c3b6"), ("StyleAligned", "stylealigned", "#b9b3a6"),
           ("InstantStyle", "instantstyle", "#a8a296"), ("SDXL LoRA\n+ ControlNet", "fulllora", "#979185"),
           ("Qwen, untrained", "qwen_q0", "#7b89a2"), ("ours", "qwen_q1", "#201963")]
MEASURES = [("raw_line_recall_t3", "line recall ↑"), ("raw_sil_iou", "silhouette IoU ↑"), ("clip_style_dedup", "CLIP style ↑"), ("palette_js", "colour-mix distance ↓")]


EXPS = [("Experiment 1: in-domain drawings (n = 19)", ("ind",)), ("Experiment 2: transfer drawings (n = 15)", ("ood", "repo"))]


def split_of(t):
    return t["content"].astype(str).str.extract(r"^(ind|ood|repo)")[0]


def main():
    tabs = {}
    for _, m, _ in METHODS:
        t = per_drawing(m).reset_index()
        t["split"] = split_of(t)
        tabs[m] = t
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.9))
    for r, (title, splits) in enumerate(EXPS):
      for ax, (col, lab) in zip(axes[r], MEASURES):
        data = [tabs[m].loc[tabs[m].split.isin(splits), col].dropna().values for _, m, _ in METHODS]
        ref = np.median(tabs["ksl_original"][col].dropna().values)
        if col != "palette_js":  # references belong to the corpus, so their colour-mix distance is ~0
            ax.axhline(ref, color="#8c8c8c", lw=0.7, ls="--", zorder=0)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.1))
        for patch, (_, _, c) in zip(bp["boxes"], METHODS):
            patch.set_facecolor(c); patch.set_edgecolor(c)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color("#6b6a64"); w.set_linewidth(0.6)
        for x, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), x) + np.random.default_rng(x).uniform(-0.18, 0.18, len(d)), d, s=2, color="#1f1f1e", alpha=0.35, zorder=3)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([m[0] for m in METHODS] if r == 1 else [], rotation=55, ha="right", fontsize=5.5)
        ax.set_title(lab, fontsize=7); ax.tick_params(axis="y", labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
      axes[r][0].annotate(("a  " if r == 0 else "b  ") + title, (0, 1.16), xycoords="axes fraction", fontsize=7.5, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.17, wspace=0.35, hspace=0.42)
    fig.savefig(ROOT / "paper/figures/fig_metric_boxes.pdf")
    fig.savefig(ROOT / "paper/figures/fig_metric_boxes.png", dpi=220)
    print("saved")


if __name__ == "__main__":
    main()
