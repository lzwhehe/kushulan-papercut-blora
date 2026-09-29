"""Shared matplotlib style for the paper's figures.

Follows the Nature-family figure guidance: sans-serif lettering in one typeface throughout (Helvetica or Arial;
here the Helvetica clone TeX Gyre Heros, with Arial/Helvetica/DejaVu Sans as fallbacks), 5-7 pt text at final
size with 8 pt bold lowercase panel letters, line weights 0.25-1 pt, white background, no top/right spines,
and colours from the paper-stock palette of the corpus (colour-blind checked pairs: indigo/gold, indigo/red).
Column widths: 89 mm single, 183 mm double.
"""
import glob
import os

import matplotlib as mpl
from matplotlib import font_manager as fm

for _f in glob.glob(os.path.expanduser("~/.local/share/fonts/*.otf")):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
INDIGO, RED, GOLD, LEAF, SLATE = "#201963", "#c52929", "#e6af25", "#4d7b35", "#7b89a2"
SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4  # inches

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros", "mathtext.it": "TeX Gyre Heros:italic", "mathtext.bf": "TeX Gyre Heros:bold",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "patch.linewidth": 0.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_label(ax, letter, x=-0.12, y=1.04):
    """8 pt bold lowercase panel letter at the top-left of an axes (Nature style)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)
