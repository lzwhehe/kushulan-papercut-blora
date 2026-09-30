"""Tables for plan v5 (Qwen-Image-Edit with cut-line pairs as the only method).

table_v5_compare.tex   Table 4: comparison with popular frameworks (same drawings), plus unconstrained generation
table_v5_canny.tex     Table 5: cut lines vs generic Canny edges as training pairs (same model and settings)
table_v5_plans.tex     Table 6: cutting plans and 200 mm SVG validation
table_v5_prereg.tex    Supplementary: pre-registered comparison Q1 vs Q0 (four endpoints, Holm)
Drawing-level means (seeds averaged); 95 % bootstrap CIs of paired differences; Wilcoxon p (exploratory
unless stated). Paper colours = colours with at least one piece in the layered analysis (outputs/layered).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from make_tables3 import per_drawing
from stats import compare, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/sections"
OURS, Q0, CANNY = "qwen_q1", "qwen_q0", "qwen_q1_canny"


def layered(method):
    f = ROOT / f"outputs/layered/{method}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["content"] = d.file.str.replace(r"_s\d*_seed\d+\.png$", "", regex=True)
    return d.groupby("content")[["colours", "pieces", "base_pieces", "layers"]].mean()


def drawing_table(method):
    d = per_drawing(method)
    if d is None:
        return None
    lay = layered(method)
    if lay is not None:
        attrs = dict(d.attrs)
        d = d.join(lay.rename(columns=lambda c: "lay_" + c), how="left")
        d.attrs.update(attrs)
    return d


def fmt(v, k):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{k}f}"


COMPARE_COLS = [("raw_line_recall_t3", 2), ("raw_sil_iou", 2), ("clip_style_dedup", 3), ("palette_js", 2), ("palette_de", 1),
                ("lay_colours", 1), ("purity", 2), ("kid", 2)]


def compare_table():
    groups = [("\\emph{Line drawing as input}", [("B-LoRA (repository workflow)", "blora"), ("StyleAligned + ControlNet", "stylealigned"),
                                               ("InstantStyle + ControlNet", "instantstyle"), ("SDXL LoRA + ControlNet", "fulllora"),
                                               ("Qwen-Image-Edit, instruction only", Q0), ("Ours (cut-line pairs)", OURS)]),
              ("\\emph{No line drawing (prompt or blank input only)}", [("SDXL, prompt only", "prompt_only"),
                                                                        ("Qwen-Image-Edit, blank input", "qwen_q0_blank"),
                                                                        ("Ours, blank input", "qwen_q1_blank")])]
    lines = []
    ref = drawing_table("ksl_original")
    lines.append("Ku Shulan originals & " + " & ".join(fmt(ref.attrs["kid"] if c == "kid" else ref[c].mean(), k) for c, k in COMPARE_COLS) + r" \\")
    for title, rows in groups:
        body = []
        for lab, m in rows:
            d = drawing_table(m)
            if d is None:
                continue
            body.append(lab + " & " + " & ".join(fmt(d.attrs["kid"] if c == "kid" else (d[c].mean() if c in d else np.nan), k)
                                                 for c, k in COMPARE_COLS) + r" \\")
        if body:
            lines += [r"\midrule", r"\multicolumn{9}{l}{" + title + r"}\\"] + body
    (OUT / "table_v5_compare.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


CANNY_ROWS = [("line recall", "raw_line_recall_t3", 2, True), ("silhouette IoU", "raw_sil_iou", 2, True),
              ("CLIP style", "clip_style_dedup", 3, True), ("palette distance $\\Delta E_{00}$", "palette_de", 1, False),
              ("colour-mix distance", "palette_js", 2, False), ("paper colours per design", "lay_colours", 1, False),
              ("colour purity", "purity", 2, True), ("chroma $C^*$", "chroma", 0, True), ("pieces per plan", "lay_pieces", 0, False)]


def paired(a, b, col):
    j = a[[col]].join(b[[col]], lsuffix="_a", rsuffix="_b").dropna()
    x, y = j[col + "_a"].values, j[col + "_b"].values
    d = x - y
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    from scipy.stats import wilcoxon
    p = wilcoxon(x, y).pvalue if not np.allclose(d, 0) else 1.0
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), p


def pstr(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def canny_table():
    o, z, c, q = (drawing_table(m) for m in ("ksl_original", Q0, CANNY, OURS))
    lines = []
    for lab, col, k, _ in CANNY_ROWS:
        dm, lo, hi, p = paired(q, c, col)
        orig = fmt(o[col].mean(), k) if col in o else "--"
        lines.append(f"{lab} & {orig} & {fmt(z[col].mean(), k)} & {fmt(c[col].mean(), k)} & {fmt(q[col].mean(), k)} & "
                     f"{dm:+.{max(k, 1)}f} [{lo:+.{max(k, 1)}f}, {hi:+.{max(k, 1)}f}] & {pstr(p)} \\\\")
    (OUT / "table_v5_canny.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def prereg_table():
    rows = []
    for m, hb in (("raw_line_recall_t3", True), ("raw_sil_iou", True), ("clip_style_dedup", True), ("palette_js", False)):
        rows.append(compare(OURS, Q0, m, hb))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.values)
    names = {"raw_line_recall_t3": "line recall", "raw_sil_iou": "silhouette IoU", "clip_style_dedup": "CLIP style", "palette_js": "colour-mix distance"}
    lines = [f"{names[r.metric]} & {r.mean_b:.3f} & {r.mean_a:.3f} & {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.r_rb:+.2f} & {pstr(r.p_holm)} \\\\"
             for _, r in df.iterrows()]
    (OUT / "table_v5_prereg.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return df


def plans_table():
    raw = pd.read_csv(ROOT / "outputs/svg_validation.csv")
    enf = pd.read_csv(ROOT / "outputs/svg_validation_enforced.csv")
    rows = [("Ku Shulan originals", "ksl_original"), ("B-LoRA", "blora"), ("StyleAligned + ControlNet", "stylealigned"),
            ("InstantStyle + ControlNet", "instantstyle"), ("SDXL LoRA + ControlNet", "fulllora"),
            ("Qwen-Image-Edit, instruction only", Q0), ("Ours", OURS)]
    lines = []
    for lab, m in rows:
        d = drawing_table(m)
        r, e = raw[raw.method == m], enf[enf.method == m]
        cells = [fmt(d["proj_line_recall_t3"].mean(), 2), fmt(d["proj_sil_iou"].mean(), 2), fmt(d["proj_clip_style_dedup"].mean(), 3),
                 fmt(d["lay_colours"].mean() if "lay_colours" in d else np.nan, 1),
                 fmt(e.pieces.median(), 0) if len(e) else "--",
                 (f"{100 * (r.pieces_below_min > 0).mean():.0f}\\,\\%" if len(r) else "--"),
                 (f"{100 * (e.pieces_below_min > 0).mean():.0f}\\,\\%" if len(e) else "--"),
                 fmt(e.smallest_mm2.mean(), 1) if len(e) else "--",
                 fmt(e.plan_line_recall.mean(), 2) if len(e) else "--"]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
        if lab == "Ku Shulan originals":
            lines.append(r"\midrule")
    (OUT / "table_v5_plans.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    compare_table()
    canny_table()
    prereg_table()
    plans_table()
