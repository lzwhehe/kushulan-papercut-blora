"""Development-set checkpoint selection for the Qwen-Image-Edit cut-line LoRA (plan v5, Fig. 6).

a  Outputs of the three epoch checkpoints on the three development drawings (seed 0).
b  Development metrics per epoch (3 drawings x 4 seeds): line recall (3 px) and palette distance, with the
   pre-registered rule (recall >= 0.90, then lowest palette distance) marked.
Also writes outputs/dev_epochs.csv. Development drawings are not part of the test set.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology

import priors
from revision_metrics import fscore

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "outputs/qwen_remote/qwen"
CDIR = ROOT / "outputs/contents_dev"
IDS = [("dev_lion", "lion"), ("dev_peach_branch", "peach branch"), ("dev_lady_holding_a_fan", "lady with a fan")]
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")


def metrics(img_path, cid):
    rgb = priors.load_rgb(img_path, 512)
    la = priors.load_rgb(CDIR / f"{cid}.png", 512)
    gt = morphology.skeletonize(la.mean(-1) < 0.85)
    _, r, _ = fscore(priors.edge_map(rgb), gt, 3)
    lab = priors.to_lab(rgb)
    fg = priors.foreground_mask(lab)
    _, de = priors.nearest_palette(lab[fg][None], PAL.lab)
    return r, float(de.mean())


def main():
    rows = []
    for e in range(3):
        for cid, _ in IDS:
            for s in range(4):
                f = DEV / f"dev_q1_epoch{e}" / f"{cid}_s_seed{s}.png"
                r, de = metrics(f, cid)
                rows.append({"epoch": e + 1, "content": cid, "seed": s, "line_recall": r, "palette_de": de})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/dev_epochs.csv", index=False)
    summ = df.groupby("epoch")[["line_recall", "palette_de"]].mean()
    print(summ.round(3))
    eligible = summ[summ.line_recall >= 0.90]
    chosen = int(eligible.palette_de.idxmin()) if len(eligible) else int(summ.line_recall.idxmax())

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(3, 4, left=0.04, right=0.60, top=0.88, bottom=0.02, wspace=0.04, hspace=0.05)
    for i, (cid, name) in enumerate(IDS):
        cells = [CDIR / f"{cid}.png"] + [DEV / f"dev_q1_epoch{e}" / f"{cid}_s_seed0.png" for e in range(3)]
        for j, p in enumerate(cells):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(Image.open(p).convert("RGB").resize((320, 320)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.6 if j == chosen else 0.3)
                sp.set_color("#c52929" if j == chosen else "#bbbbbb")
            if i == 0:
                ax.set_title("drawing" if j == 0 else f"epoch {j}" + (" (selected)" if j == chosen else ""), fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
    ep = summ.index.values
    g2 = fig.add_gridspec(2, 1, left=0.70, right=0.98, top=0.88, bottom=0.14, hspace=0.35)
    a1 = fig.add_subplot(g2[0]); a2 = fig.add_subplot(g2[1], sharex=a1)
    a1.plot(ep, summ.line_recall, "o-", color="#201963", lw=1.2, ms=4)
    a1.axhline(0.90, color="#201963", lw=0.6, ls=":"); a1.text(1.05, 0.903, "threshold 0.90", fontsize=6, color="#201963")
    a1.set_ylabel("line recall", fontsize=7)
    a2.plot(ep, summ.palette_de, "s-", color="#c52929", lw=1.2, ms=4)
    a2.set_ylabel("palette dist. ΔE$_{00}$", fontsize=7)
    a2.set_xticks(ep); a2.set_xlabel("epoch (1,028 steps each)", fontsize=7)
    for a in (a1, a2):
        a.tick_params(labelsize=6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.axvline(chosen, color="#c52929", lw=0.6, alpha=0.4)
    plt.setp(a1.get_xticklabels(), visible=False)
    fig.text(0.04, 0.955, "a  outputs of each checkpoint (seed 0)", fontsize=7, weight="bold")
    fig.text(0.66, 0.955, "b  development set (3 drawings × 4 seeds)", fontsize=7, weight="bold")
    fig.savefig(ROOT / "paper/figures/fig_dev_epochs.pdf")
    fig.savefig(ROOT / "paper/figures/fig_dev_epochs.png", dpi=220)
    print("chosen epoch", chosen)


if __name__ == "__main__":
    main()
