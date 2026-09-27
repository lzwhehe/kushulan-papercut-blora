"""Fig. 2: the Ku Shulan craft space and validation of the craft metrics."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import figures as F
import priors

ROOT = F.ROOT


def perturb(p):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    rgb = priors.load_rgb(ROOT / p, 512)
    rng = np.random.default_rng(abs(hash(p)) % 2**32)
    lab = priors.to_lab(rgb)
    fg = priors.foreground_mask(lab)
    y = np.linspace(0, 1, 512)[:, None, None]
    shade = np.where(fg[..., None], np.clip(rgb * (0.65 + 0.6 * y), 0, 1), rgb)  # vertical shading on the figure only
    out = []
    for name, x in [("original", rgb), ("blurred", gaussian_filter(rgb, (3, 3, 0))),
                    ("noisy", np.clip(rgb + rng.normal(0, 0.04, rgb.shape), 0, 1)), ("shaded", shade)]:
        m = priors.craft_metrics(x.astype(np.float32), pal)
        m.update(condition=name, path=p)
        out.append(m)
    return out


if __name__ == "__main__":
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    rng = np.random.default_rng(0)
    paths = [it["path"] for it in split["train"]]
    sample = [paths[i] for i in rng.choice(len(paths), 60, replace=False)]
    with ProcessPoolExecutor(24) as ex:
        rows = [r for rs in ex.map(perturb, sample) for r in rs]
    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(ROOT / "outputs/data/metric_validation.csv", index=False)
    print(ctrl.groupby("condition")[["purity", "palette_de", "cut_residual", "fragments"]].median().round(3))

    corpus = pd.read_csv(ROOT / "outputs/data/corpus_stats.csv")
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 5], hspace=0.9, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    F.palette_panel(ax, pal)
    ax.set_title("a  Ku Shulan palette (16 colours, CIELAB k-means on 257 training works)", loc="left", fontsize=7)
    order = ["original", "blurred", "noisy", "shaded"]
    cols = ["#c0392b", "#7f8c8d", "#95a5a6", "#bdc3c7"]
    for i, (m, lab) in enumerate([("purity", "colour purity ↑"), ("palette_de", "palette ΔE00 ↓"),
                                  ("cut_residual", "cut residual ΔE00 ↓")]):
        ax = fig.add_subplot(gs[1, i])
        data = [ctrl[ctrl.condition == c][m].values for c in order]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=0.8))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c), patch.set_linewidth(0.5)
        ax.set_xticks(range(1, 5), order, rotation=35, fontsize=6)
        ax.set_title(("b  " if i == 0 else "") + lab, fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[1, 3])
    rep = corpus[corpus.subset == "representative"]
    cats = ["person", "animals", "plants", "daily", "frame", "window decoration"]
    names = ["figure", "animal", "plant", "object", "border", "window"]
    for c, n in zip(cats, names):
        d = rep[rep.category == c]
        ax.scatter(d["symmetry"], d["purity"], s=5, label=n, alpha=0.8, lw=0)
    ax.set_xlabel("left–right symmetry", fontsize=6.5)
    ax.set_ylabel("colour purity", fontsize=6.5)
    ax.set_title("c  per category", fontsize=7, loc="left")
    ax.legend(fontsize=5, frameon=False, handletextpad=0.1, loc="lower left", markerscale=1.5)
    ax.tick_params(labelsize=6)
    fig.savefig(ROOT / "paper/figures/fig2_craft_space.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper/figures/fig2_craft_space.png", bbox_inches="tight", dpi=200)
