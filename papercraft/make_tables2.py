"""Tables 2, 3, 4 and 6 of the npj-style manuscript (ablations and sensitivity)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
OUT = ROOT / "paper/sections"


def load(m):
    """Per-drawing means (review #09/#10: the drawing is the statistical unit)."""
    from make_tables3 import per_drawing
    df = per_drawing(m)
    if df is None:
        return None
    kid = df.attrs["kid"]
    df = df.reset_index()
    df["kid"] = kid
    return df


def fmt(v, dec):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dec}f}"


def table(rows, cols, fname, base=None):
    frames = {m: load(m) for _, m in rows}
    b = frames.get(base) if base else None
    lines = []
    for lab, m in rows:
        df = frames[m]
        if df is None:
            lines.append(lab + " & " + " & ".join([r"\pending{}"] * len(cols)) + r" \\")
            continue
        cells = []
        for c, dec in cols:
            if c not in df or df[c].notna().sum() < 0.5 * len(df):  # undefined for mostly blank outputs
                cells.append("--")
                continue
            s = fmt(float(df[c].mean()), dec)
            if b is not None and m != base and c not in ("kid", "fg_frac") and c in b:
                j = df.merge(b, on="content", suffixes=("", "_b"))[[c, c + "_b"]].dropna()
                thr = 0.5 if c in ("palette_de", "cut_residual") else (1.0 if c in ("chroma", "raw_decoration_density") else 0.01)
                if len(j) > 5 and not np.allclose(j[c], j[c + "_b"]) and abs(j[c].mean() - j[c + "_b"].mean()) >= thr:
                    p = wilcoxon(j[c], j[c + "_b"]).pvalue
                    s += r"$^{***}$" if p < 1e-3 else r"$^{**}$" if p < 1e-2 else r"$^{*}$" if p < 0.05 else ""
            cells.append(s)
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (OUT / fname).write_text("\n".join(lines) + "\n")
    print(fname); print("\n".join(lines))


C = [("palette_de", 1), ("purity", 2), ("cut_residual", 1), ("raw_line_recall_t3", 2), ("raw_edge_prec_t3", 2),
     ("raw_sil_iou", 2), ("raw_decoration_density", 1), ("clip_style_dedup", 3), ("palette_js", 2), ("chroma", 0)]
if __name__ == "__main__":
    table([("CutCraft (full)", "cutcraft"), ("w/o palette term $E_{\\mathrm{pal}}$", "abl_no_pal"),
           ("w/o flatness term $E_{\\mathrm{flat}}$", "abl_no_flat"), ("w/o line-support term $E_{\\mathrm{edge}}$", "abl_no_edge"),
           ("w/o paper-ground term $E_{\\mathrm{ground}}$", "cutcraft_noground"), ("w/o CraftGuide (cut-line block)", "cutline"),
           ("w/o cut-line training (collection block + CraftGuide)", "coll_guide")],
          C, "table_components.tex", base="cutcraft")
    table([("Collection block (coll)", "coll"), ("coll + self craft regulariser", "var_coll_self"),
           ("coll + self + foreign craft energy", "var_energy"), ("Projection self-distillation", "var_pd"),
           ("Cut-line block", "cutline"), ("Cut-line block + self craft regulariser", "cutline_self"),
           ("Cut-line block, dense maps only", "cl_dense"), ("Cut-line block, coarse maps only", "cl_coarse"),
           ("Canny edges of the image instead of cut lines", "cl_canny"),
           ("Cut-line block, strict split (235 images)", "cutline_strict"),
           ("Conv.\\ LoRA (all attention layers)", "fulllora"), ("Conv.\\ LoRA + cut-line training", "fulllora_cl")],
          [("fg_frac", 2), ("palette_de", 1), ("raw_line_recall_t3", 2), ("raw_sil_iou", 2), ("clip_style_dedup", 3), ("clip_text", 3),
           ("palette_js", 2), ("chroma", 0), ("kid", 2)], "table_variants.tex", base="cutline")
    table([("$\\eta=0$ (no guidance)", "cutline"), ("$\\eta=0.3$", "sens_g0.3"), ("$\\eta=0.6$", "sens_g0.6"),
           ("$\\eta=1.0$ (selected)", "cutcraft"), ("$\\eta=2.0$", "sens_g2.0")],
          [("palette_de", 1), ("cut_residual", 1), ("raw_line_recall_t3", 2), ("raw_edge_prec_t3", 2), ("raw_sil_iou", 2),
           ("clip_style_dedup", 3), ("kid", 2), ("chroma", 0)], "table_eta.tex", base="cutcraft")
    table([("$s=0.4$", "sens_cn0.4"), ("$s=0.6$ (selected)", "cutcraft"), ("$s=0.8$", "sens_cn0.8")],
          [("palette_de", 1), ("cut_residual", 1), ("raw_line_recall_t3", 2), ("raw_edge_prec_t3", 2), ("raw_sil_iou", 2),
           ("clip_style_dedup", 3), ("kid", 2), ("chroma", 0)], "table_cn.tex", base="cutcraft")
