"""Supplementary Table S1: every configuration explored on the development set."""
import json
from pathlib import Path

import pandas as pd

O = Path(__file__).resolve().parents[1] / "outputs"
ROWS = [("coll, ControlNet 0.4", "dev/cn0.4"), ("coll, ControlNet 0.6", "dev/cn0.6"), ("coll, ControlNet 0.8", "dev/cn0.8"),
        ("energy regulariser (foreign)", "dev/guide0"), ("energy regulariser + guide 0.4", "dev/guide0.4"),
        ("projection self-distillation", "dev/cutcraft_pd_g0"),
        ("prompt only", "dev/prompt_cn"), ("coll + guide 0.6", "dev3/coll_g0.6"),
        ("cut-line, ControlNet 0.6", "dev3/cutline_cn0.6"), ("cut-line, ControlNet 0.8", "dev3/cutline_cn0.8"),
        ("cut-line + guide 0.3", "dev3/cutline_g0.3"), ("cut-line + guide 0.6", "dev3/cutline_g0.6"),
        ("cut-line + guide 1.0 (selected)", "dev3/cutline_g1.0"), ("cut-line + guide 2.0", "dev3/cutline_g2.0")]
out, lines = [], []
for name, d in ROWS:
    if not (O / d / "metrics.csv").exists():
        continue
    x = pd.read_csv(O / d / "metrics.csv")
    k = json.loads((O / d / "metrics_kid.json").read_text())
    g = lambda c: x[c].mean() if c in x else float("nan")  # blank outputs have no foreground to measure
    r = dict(setting=name, palette_de=g("palette_de"), purity=g("purity"), cut_residual=g("cut_residual"),
             fragments=g("fragments"), fg=g("fg_frac"), edge_f1=g("edge_f1"), sil_iou=g("sil_iou"),
             clip_style=g("clip_style"), clip_text=g("clip_text"), kid=k["clip_kid_x1000"])
    out.append(r)
    f = lambda v, d: "--" if v != v else f"{v:.{d}f}"
    lines.append(f"{name} & {f(r['palette_de'], 1)} & {f(r['purity'], 2)} & {f(r['cut_residual'], 1)} & "
                 f"{f(r['fragments'], 0)} & {f(r['fg'], 2)} & {f(r['edge_f1'], 2)} & {f(r['sil_iou'], 2)} & "
                 f"{f(r['clip_style'], 3)} & {f(r['kid'], 2)} \\\\")
pd.DataFrame(out).to_csv(O / "dev_table.csv", index=False)
(O.parent / "paper/sections/table_dev.tex").write_text("\n".join(lines) + "\n")
print(pd.DataFrame(out).round(3).to_string(index=False))
