"""Paper figures from generated images and metric tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pubstyle  # noqa: F401  shared publication style (fonts, sizes, spines)
import numpy as np
import pandas as pd
from PIL import Image

import priors

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
GEN = ROOT / "outputs/gen"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
INK = "#222222"


def _img(path, size=384):
    return Image.fromarray((priors.load_rgb(path, size) * 255).astype(np.uint8))


def grid(rows, cols, out, cell=1.25, row_labels=None, col_labels=None):
    """rows: list of lists of image paths (or PIL images or None)."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(cell * len(cols), cell * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, p in enumerate(r):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(p is not None)
                s.set_linewidth(0.4)
                s.set_edgecolor("#b0b0b0")
            if p is None:
                continue
            ax.imshow(p if isinstance(p, Image.Image) else _img(p))
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=6.5, pad=3)
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], fontsize=6.5)
    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.03, right=0.995, top=0.95, bottom=0.005)
    kw = {"pil_kwargs": {"quality": 90}} if str(out).endswith(".jpg") else {}
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def dotplot(table: pd.DataFrame, metrics, labels, out, order=None, ref=None, highlight=None):
    """One small panel per metric: mean and 95 % CI per method."""
    order = order or list(table["method"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.55 * len(metrics), 0.22 * len(order) + 0.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, m, lab in zip(axes, metrics, labels):
        for yy, meth in zip(y, order):
            r = table[table["method"] == meth].iloc[0]
            lo, hi = [float(v) for v in r[m + "_ci"].strip("[]").split(",")]
            c = "#c0392b" if highlight and meth in highlight else "#34495e"
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.2)
            ax.plot(r[m], yy, "o", color=c, ms=3)
        if ref is not None and m in ref:
            ax.axvline(ref[m], color="#27ae60", ls="--", lw=0.8)
        ax.set_title(lab, fontsize=6.5)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=6.5)
    plt.tight_layout(w_pad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def palette_panel(ax, pal: priors.Palette):
    rgb = pal.rgb
    x = 0
    for c, w in zip(rgb, pal.weight):
        ax.add_patch(plt.Rectangle((x, 0), w, 1, color=c, lw=0))
        x += w
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("share of foreground pixels", fontsize=6.5)


def layers_figure(img_path, out, pal: priors.Palette, size=512, ground=None, ncol=4):
    rgb = priors.load_rgb(img_path, size)
    proj, labels, _, st = priors.cut_project(rgb, pal, ground=ground)
    pal_lab = pal.lab_with_white
    present = [k for k in np.argsort(pal_lab[:, 0]) if k != len(pal_lab) - 1 and (labels == k).mean() > 1e-3]
    n = len(present)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7.2, 7.2 * max(nrow, 2) / (ncol + 4)))
    gs = fig.add_gridspec(max(nrow, 2), ncol + 4, wspace=0.05, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0:2]); ax.imshow(rgb); ax.set_title("a  generated", fontsize=7, loc="left")
    ax2 = fig.add_subplot(gs[:, 2:4]); ax2.imshow(proj)
    ax2.set_title(f"b  cut projection ({n} sheets)", fontsize=7, loc="left")
    cols = np.clip(priors.color.lab2rgb(pal_lab[None])[0], 0, 1)
    fgpix = max((labels != len(pal_lab) - 1).sum(), 1)
    axes = [ax, ax2]
    for i, k in enumerate(present):
        a_ = fig.add_subplot(gs[i // ncol, 4 + i % ncol])
        m = labels == k
        sheet = np.ones_like(rgb)
        sheet[m] = cols[k]
        a_.imshow(sheet)
        hexc = "#%02x%02x%02x" % tuple((cols[k] * 255).astype(int))
        a_.set_title(f"{hexc} {100 * m.sum() / fgpix:.0f}%", fontsize=5, pad=1)
        axes.append(a_)
    for a_ in axes:
        a_.set_xticks([]), a_.set_yticks([])
        for s in a_.spines.values():
            s.set_linewidth(0.3)
    kw = {"pil_kwargs": {"quality": 92}} if str(out).endswith(".jpg") else {}
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=250, **kw)
    plt.close(fig)
    return st
