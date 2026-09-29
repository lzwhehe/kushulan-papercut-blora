"""Plug-in figure: each host model before (hollow) and after (filled) adding a craft component.

Small multiples, one panel per primary endpoint, shared host axis; every panel is oriented so that right is
better (colour-mix distance axis inverted). Colours: reference categorical slots 1-2 (blue, orange); the
component is also encoded by marker shape so identity is not colour-alone. Lines join before/after.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pubstyle  # noqa: F401  shared publication style (fonts, sizes, spines)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#201963", "#e6af25", "#c52929", "#1f1f1e", "#6b6a64", "#e6e5df"  # paper-stock palette: indigo, gold, red
PANELS = [("raw_line_recall_t3", "Line recall", False), ("raw_sil_iou", "Silhouette IoU", False),
          ("clip_style_dedup", "CLIP style", False), ("palette_js", "Colour-mix distance", True)]
STYLE = {"Scissor-path training": (BLUE, "o"), "Scissor-path maps vs generic edges": (AQUA, "s"), "Craft guidance": (ORANGE, "D")}


def main():
    df = pd.read_csv(ROOT / "outputs/plugin_matrix.csv")
    labels = [f"{h}{' †' if p else ''}" for h, p in zip(df.host, df.prespecified)]
    y = list(range(len(df)))[::-1]
    order = list(dict.fromkeys(df.component))
    gap = [yy - 0.6 * order.index(c) for yy, c in zip(y, df.component)]
    plt.rcParams.update({"font.size": 7, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.9), sharey=True)
    for ax, (m, title, invert) in zip(axes, PANELS):
        for yy, (_, r) in zip(gap, df.iterrows()):
            col, mk = STYLE[r.component]
            b, a = r[f"{m}_before"], r[f"{m}_after"]
            ax.plot([b, a], [yy, yy], color=GRID if abs(a - b) < 1e-9 else col, lw=2, alpha=0.45, zorder=1,
                    solid_capstyle="round")
            ax.scatter([b], [yy], s=34, facecolor="white", edgecolor=MUTED, lw=1.2, marker=mk, zorder=2)
            # same rule as table_plugin.tex: CI excludes zero and the change is not negligible
            sig = (r[f"{m}_lo"] > 0 or r[f"{m}_hi"] < 0) and abs(a - b) >= 0.01
            ax.scatter([a], [yy], s=40, color=col, marker=mk, zorder=3, edgecolor="white", lw=1.0,
                       alpha=1.0 if sig else 0.45)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if invert:
            ax.invert_xaxis()
        ax.annotate("better →", xy=(1, -0.13), xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)
    axes[0].set_yticks(gap); axes[0].set_yticklabels(labels)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="-", lw=2, ms=6, label=k) for k, (c, mk) in STYLE.items()]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=6, label="host model without the component"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 0.995))
    fig.text(0.01, 0.005, "† comparison pre-specified before test generation. Faded markers: 95% CI of the change includes zero, or |change| < 0.01. Colour-mix axis reversed.",
             fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.79, bottom=0.11, wspace=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper/figures/fig_plugin.{ext}", dpi=220)
    print("saved")


if __name__ == "__main__":
    main()
