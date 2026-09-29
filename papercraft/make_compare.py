"""Compact comparison with popular frameworks (plan v3.1, part B).

Writes paper/sections/table_compare.tex (drawing-level means) and outputs/compare_stats.csv:
CutCraft-SDXL vs each baseline on the four primary endpoints (Holm over all of them; B-LoRA, InstantStyle +
guide and conventional LoRA + guide were in the original primary family, StyleAligned and Qwen zero-shot
were added by plan v3 before they were scored) and CutCraft-Qwen vs the same baselines (exploratory).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from make_tables3 import per_drawing
from stats import compare, holm

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("Ku Shulan originals", "ksl_original"), (None, None),
        ("B-LoRA (repository workflow)", "blora"), ("StyleAligned + ControlNet", "stylealigned"),
        ("InstantStyle + ControlNet", "instantstyle"), ("Conv.\\ LoRA + ControlNet", "fulllora"),
        ("Qwen-Image-Edit-2511 (zero-shot)", "qwen_q0"), (None, None),
        ("CutCraft-SDXL", "cutcraft"), ("CutCraft-Qwen", "qwen_q1")]
COLS = [("raw_line_recall_t3", 2), ("raw_sil_iou", 2), ("clip_style_dedup", 3), ("palette_js", 2), ("edge_f1", 2), ("palette_de", 1),
        ("kid", 2), ("content_clip", 3), ("style_gram", 3), ("clip_s", 3), ("lpips_content", 2)]
PRIMARY = [("raw_line_recall_t3", True), ("raw_sil_iou", True), ("clip_style_dedup", True), ("palette_js", False)]
BASELINES = ["blora", "stylealigned", "instantstyle", "fulllora", "qwen_q0"]


def main():
    lines = []
    for lab, m in ROWS:
        if m is None:
            lines.append(r"\midrule"); continue
        d = per_drawing(m)
        cells = [f"{d.attrs['kid']:.2f}" if c == "kid" else (f"{d[c].mean():.{k}f}" if c in d else "--") for c, k in COLS]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (ROOT / "paper/sections/table_compare.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    rows = []
    for ours, fam in (("cutcraft", "primary"), ("qwen_q1", "exploratory")):
        for b in BASELINES:
            for m, hb in PRIMARY:
                r = compare(ours, b, m, hb); r["family"] = fam; rows.append(r)
    for m, hb in PRIMARY:  # pre-registered Qwen comparison
        r = compare("qwen_q1", "qwen_q0", m, hb); r["family"] = "qwen"; rows.append(r)
    df = pd.DataFrame(rows)
    prim = df.family == "primary"
    df.loc[prim, "p_holm"] = holm(df.loc[prim, "p"].values)
    q = df.family == "qwen"
    df.loc[q, "p_holm"] = holm(df.loc[q, "p"].values)
    df.to_csv(ROOT / "outputs/compare_stats.csv", index=False)
    names = {"cutcraft": "CutCraft-SDXL", "qwen_q1": "CutCraft-Qwen", "blora": "B-LoRA", "stylealigned": "StyleAligned",
             "instantstyle": "InstantStyle", "fulllora": "Conv.\\ LoRA", "qwen_q0": "Qwen zero-shot (Q0)"}
    mn = {"raw_line_recall_t3": "line recall", "raw_sil_iou": "silhouette IoU", "clip_style_dedup": "CLIP style", "palette_js": "colour-mix"}
    lines = []
    for fam, lab in (("primary", "second primary family (20 tests, Holm)"), ("qwen", "pre-registered Q1 vs Q0 (4 tests, Holm)"), ("exploratory", "exploratory")):
        lines.append("\\midrule\n\\multicolumn{7}{l}{\\emph{" + lab + "}}\\\\")
        for _, r in df[df.family == fam].iterrows():
            ph = ("$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}") if fam != "exploratory" else f"{r.p:.3f} (unc.)"
            lines.append(f"{names[r.a]} vs {names[r.b]} & {mn[r.metric]} & {r.mean_a:.3f} & {r.mean_b:.3f} & {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.r_rb:+.2f} & {ph} \\\\")
    (ROOT / "paper/sections/table_compare_stats.tex").write_text("\n".join(lines[1:] if lines[0].startswith("\\midrule") else lines) + "\n")
    with pd.option_context("display.width", 200):
        print(df[["family", "a", "b", "metric", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "r_rb", "p", "p_holm", "better"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
