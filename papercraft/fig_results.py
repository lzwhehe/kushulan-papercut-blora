"""Results figures: qualitative comparison, metric dot-plot, cut layers, failure modes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import figures as F
import priors

ROOT = F.ROOT
G = ROOT / "outputs/gen"
FIG = ROOT / "paper/figures"
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")


def out_img(method, cid, seed=0, style=None):
    s = "" if style is None else str(style)
    p = G / method / f"{cid}_s{s}_seed{seed}.png"
    return p if p.exists() else None


def ground_of(cid):
    la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
    return priors.ground_from_lineart(la.mean(-1) < 0.85)


def projected(p, cid):
    rgb = priors.load_rgb(p, 512)
    return Image.fromarray((priors.cut_project(rgb, PAL, ground=ground_of(cid))[0] * 255).astype(np.uint8))


def qualitative(ids, out, seed=0):
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    ref = {"ind_" + t["id"]: ROOT / t["reference"] for t in split["test_indomain"]}
    cols = ["line drawing", "B-LoRA", "prompt only", "coll", "cut-line", "CutCraft", "CutCraft + cut proj.",
            "Ku Shulan original"]
    rows = []
    for cid in ids:
        cc = out_img("cutcraft", cid, seed)
        rows.append([ROOT / f"outputs/contents/{cid}.png", out_img("blora", cid, 0, 0), out_img("prompt_cn", cid, seed),
                     out_img("coll", cid, seed), out_img("cutline", cid, seed), cc,
                     projected(cc, cid) if cc else None, ref.get(cid)])
    F.grid(rows, cols, out, cell=0.98, col_labels=cols)


def failure_modes(out):
    """Dev-set outputs of the training variants (lion, peach, lady; seed 1)."""
    D = ROOT / "outputs"
    ids = ["dev_lion", "dev_peach_branch", "dev_lady_holding_a_fan"]
    cols = [("dev/cn0.6", "coll"), ("dev/guide0", "energy reg. (foreign)"),
            ("dev/cutcraft_pd_g0.2", "projection self-distill."), ("dev3/cutline_cn0.6", "cut-line"),
            ("dev3/cutline_g1.0", "cut-line + CraftGuide")]
    rows = [[D / "contents_dev" / f"{i}.png"] + [D / c / f"{i}_s_seed1.png" for c, _ in cols] for i in ids]
    labels = ["line drawing"] + [l for _, l in cols]
    F.grid(rows, labels, out, cell=1.15, col_labels=labels)


def layers(cid, out, seed=0):
    return F.layers_figure(out_img("cutcraft", cid, seed), out, PAL, ground=ground_of(cid))


if __name__ == "__main__":
    what = sys.argv[1:] or ["qual", "fail"]
    if "qual" in what:
        qualitative(["repo_fish", "repo_bird", "repo_motif", "ood_rooster", "ood_teapot", "ood_rabbit",
                     "ind_动物-18", "ind_日常-25"], FIG / "fig4_qualitative.png")
    if "fail" in what:
        failure_modes(FIG / "fig3_training_variants.png")
