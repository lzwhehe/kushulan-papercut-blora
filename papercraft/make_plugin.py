"""Plug-in matrix (plan v3.1, part A): each craft component added to several host models.

For every (component, host) pair: drawing-level means before and after, paired difference with a 95 %
bootstrap CI (10,000 resamples of drawings) and the Wilcoxon p (uncorrected; which rows were pre-specified
is marked). Writes outputs/plugin_matrix.csv and paper/sections/table_plugin.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from make_tables3 import per_drawing
from stats import compare

ROOT = Path(__file__).resolve().parents[1]
TO, DAGGER = r"$\to$", r"$^\dagger$"

ROWS = [  # component, host label, before, after, pre-specified?
    ("Scissor-path training", "SDXL style block", "coll", "cutline", False),
    ("Scissor-path training", "Conventional LoRA", "fulllora", "fulllora_cl", False),
    ("Scissor-path maps vs generic edges", "SDXL style block", "cl_canny", "cutline", False),
    ("Craft guidance", "B-LoRA", "blora", "blora_guide", False),
    ("Craft guidance", "InstantStyle", "instantstyle", "instantstyle_guide", False),
    ("Craft guidance", "Conventional LoRA", "fulllora", "fulllora_guide", False),
    ("Craft guidance", "SDXL cut-line block", "cutline", "cutcraft", False),
]
METRICS = [("raw_line_recall_t3", True, 2), ("raw_sil_iou", True, 2), ("clip_style_dedup", True, 3),
           ("palette_js", False, 2), ("palette_de", False, 1)]


def main():
    recs = []
    for comp, host, b, a, pre in ROWS:
        rec = {"component": comp, "host": host, "before": b, "after": a, "prespecified": pre,
               "kid_before": per_drawing(b).attrs["kid"], "kid_after": per_drawing(a).attrs["kid"]}
        for m, hb, _ in METRICS:
            r = compare(a, b, m, hb)
            rec.update({f"{m}_before": r["mean_b"], f"{m}_after": r["mean_a"], f"{m}_diff": r["diff"],
                        f"{m}_lo": r["ci_lo"], f"{m}_hi": r["ci_hi"], f"{m}_p": r["p"],
                        f"{m}_improved": r["better"] == "a" and (r["ci_lo"] > 0 or r["ci_hi"] < 0)})
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / "outputs/plugin_matrix.csv", index=False)
    lines, prev = [], None
    for _, r in df.iterrows():
        if prev is not None and r.component != prev:
            lines.append(r"\midrule")
        prev = r.component
        cells = []
        for m, hb, dec in METRICS:
            d = r[f"{m}_diff"]
            # arrow only if the 95 % CI excludes zero and the change is not negligible
            sig = (r[f"{m}_lo"] > 0 or r[f"{m}_hi"] < 0) and abs(d) >= (0.5 if m == "palette_de" else 0.01)
            good = (d > 0) == hb
            mark = (r"$\uparrow$" if good else r"$\downarrow$") if sig else r"$\approx$"
            cells.append(f"{r[f'{m}_before']:.{dec}f}" + TO + f"{r[f'{m}_after']:.{dec}f} {mark}")
        cells.append(f"{r.kid_before:.2f}" + TO + f"{r.kid_after:.2f}")
        dagger = DAGGER if r.prespecified else ""
        lines.append(f"{r.component} & {r.host}{dagger} & " + " & ".join(cells) + r" \\")
    # print the component name only on the first row of each block
    out, seen = [], set()
    for l in lines:
        if l == r"\midrule":
            out.append(l); continue
        comp = l.split(" & ")[0]
        out.append(("" if comp in seen else comp) + l[len(comp):])
        seen.add(comp)
    (ROOT / "paper/sections/table_plugin.tex").write_text("\n".join(out) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["component", "host"] + [f"{m}_{s}" for m, _, _ in METRICS for s in ("before", "after", "diff")] + ["kid_before", "kid_after"]
        print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
