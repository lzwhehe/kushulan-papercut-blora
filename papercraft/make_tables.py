"""LaTeX tables for the paper from per-image metric files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
OUT = ROOT / "paper/sections"

COLS = [("palette_de", r"$\Delta E_{\mathrm{pal}}\downarrow$", -1, 1), ("purity", r"pur.$\uparrow$", 1, 2),
        ("cut_residual", r"res.$\downarrow$", -1, 1), ("fragments", r"frag.$\downarrow$", -1, 0),
        ("edge_f1", r"F1$\uparrow$", 1, 2), ("sil_iou", r"IoU$\uparrow$", 1, 2),
        ("clip_style", r"style$\uparrow$", 1, 3), ("kid", r"KID$\downarrow$", -1, 2),
        ("palette_js", r"mix$\downarrow$", -1, 2), ("chroma", r"$C^*$", 1, 0),
        ("dino_styleref", r"leak$\downarrow$", -1, 2), ("dino_gt", r"GT$\uparrow$", 1, 2)]
BY_CONSTRUCTION = {"palette_de", "purity", "cut_residual", "fragments"}


def load(method, projected=False, subset=None):
    d = G / method
    df = pd.read_csv(d / ("metrics_projected.csv" if projected else "metrics.csv"))
    if subset:
        df = df[df["set"].isin(subset)]
    kid = json.loads((d / ("metrics_projected_kid.json" if projected else "metrics_kid.json")).read_text())
    num = [c for c, *_ in COLS if c in df]
    # pair key: content + seed for ControlNet methods; content + style reference for B-LoRA (seed 0)
    df["pair"] = df["seed"].astype(str) if df["style"].isna().all() else df["style"].astype(int).astype(str)
    agg = df.groupby(["content", "pair"], as_index=False)[num].mean()
    agg["kid"] = kid["clip_kid_x1000"]
    return agg


def cell(df, base, col, dec, sign, show_p):
    if col not in df or df[col].isna().all():
        return "--"
    v = df[col].mean()
    s = f"{v:.{dec}f}"
    if show_p and base is not None and col != "kid" and col in base:
        j = df.merge(base, on=["content", "pair"], suffixes=("", "_b"))[[col, col + "_b"]].dropna()
        if len(j) > 5 and not np.allclose(j[col], j[col + "_b"]):
            p = wilcoxon(j[col], j[col + "_b"]).pvalue
            if p < 0.001:
                s += r"$^{***}$"
            elif p < 0.01:
                s += r"$^{**}$"
            elif p < 0.05:
                s += r"$^{*}$"
    return s


def table(rows, fname, base_key=None, subset=None):
    lines = []
    frames = {}
    for label, method, proj in rows:
        if label.startswith("\\midrule"):
            lines.append(r"\midrule")
            continue
        try:
            frames[label] = load(method, proj, subset)
        except FileNotFoundError:
            frames[label] = None
    base = frames.get(base_key)
    for label, method, proj in rows:
        if label.startswith("\\midrule"):
            continue
        df = frames[label]
        if df is None:
            lines.append(label + " & " + " & ".join(["n/a"] * len(COLS)) + r" \\")
            continue
        cells = []
        for col, _, sign, dec in COLS:
            if proj and col in BY_CONSTRUCTION:
                cells.append(r"\textit{proj.}")
            else:
                cells.append(cell(df, base, col, dec, sign, label != base_key and method != "ksl_original"))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    # keep \midrule markers in order
    out, it = [], iter(lines)
    k = 0
    for label, *_ in rows:
        if label.startswith("\\midrule"):
            out.append(r"\midrule")
        else:
            out.append([l for l in lines if l.startswith(label + " &")][0])
    (OUT / fname).write_text("\n".join(out) + "\n")
    return frames


if __name__ == "__main__":
    main_rows = [
        ("Ku Shulan originals", "ksl_original", False),
        ("\\midrule", None, None),
        ("B-LoRA (repository workflow)", "blora", False),
        ("B-LoRA + CraftGuide", "blora_guide", False),
        ("B-LoRA style block + ControlNet", "blora_style_cn", False),
        ("Prompt only + ControlNet", "prompt_cn", False),
        ("Collection block (coll)", "coll", False),
        ("Cut-line block", "cutline", False),
        ("CutCraft (cut-line + CraftGuide)", "cutcraft", False),
        ("\\midrule", None, None),
        ("B-LoRA + cut projection", "blora", True),
        ("Prompt only + cut projection", "prompt_cn", True),
        ("Cut-line block + cut projection", "cutline", True),
        ("CutCraft + cut projection", "cutcraft", True),
    ]
    table(main_rows, "table_main.tex", base_key="B-LoRA (repository workflow)")
    abl_rows = [
        ("Cut-line block", "cutline", False),
        ("+ self craft regulariser", "cutline_self", False),
        ("+ CraftGuide (= CutCraft)", "cutcraft", False),
        ("+ CraftGuide without ground term", "cutcraft_noground", False),
        ("Collection block + CraftGuide", "coll_guide", False),
    ]
    table(abl_rows, "table_ablation.tex", base_key="Cut-line block")
    for sub, name in ((["indomain"], "table_indomain.tex"), (["repo", "ood"], "table_ood.tex")):
        table(main_rows[2:], name, base_key="B-LoRA (repository workflow)", subset=sub)
    print(open(OUT / "table_main.tex").read())
    print(open(OUT / "table_ablation.tex").read())
