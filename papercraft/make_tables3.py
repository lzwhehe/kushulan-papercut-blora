"""Revised result tables (review #03, #09, #10, #33): drawing-level means, line recall as primary structure
endpoint, de-duplicated CLIP style, separate table for the exported plans and for the primary statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
OUT = ROOT / "paper/sections"


def per_drawing(method, subset=None):
    """Per-drawing means of all per-image scores (seeds / style references averaged first)."""
    d = G / method
    if not (d / "rev_metrics.csv").exists():
        return None
    a = pd.read_csv(d / "metrics.csv")
    r = pd.read_csv(d / "rev_metrics.csv")
    p = pd.read_csv(d / "metrics_projected.csv")[["file", "dino_gt", "fragments"]].rename(
        columns={"dino_gt": "proj_dino_gt", "fragments": "proj_fragments"})
    df = a.merge(r.drop(columns=[c for c in ("content", "style", "seed", "set") if c in r]), on="file").merge(p, on="file", how="left")
    if subset:
        df = df[df["set"].isin(subset)]
    blank = df["fg_frac"] < 0.02  # colour statistics undefined for (near-)blank outputs
    for c in ("palette_de", "purity", "cut_residual", "fragments", "palette_js", "chroma"):
        df.loc[blank, c] = np.nan
    num = df.select_dtypes("number").columns.drop(["seed", "style", "nn_train_idx"], errors="ignore")
    out = df.groupby("content")[num].mean()
    out.attrs["kid"] = json.loads((d / "metrics_kid.json").read_text())["clip_kid_x1000"]
    kp = d / "metrics_projected_kid.json"
    out.attrs["kid_proj"] = json.loads(kp.read_text())["clip_kid_x1000"] if kp.exists() else np.nan
    out.attrs["blank_share"] = float(blank.mean())
    return out


def fmt(v, dec):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{dec}f}"


def write(rows, cols, fname, subset=None):
    lines = []
    for lab, m in rows:
        if lab == "midrule":
            lines.append(r"\midrule")
            continue
        df = per_drawing(m, subset)
        if df is None:
            lines.append(lab + " & " + " & ".join([r"\pending{}"] * len(cols)) + r" \\")
            continue
        cells = []
        for c, dec in cols:
            if c in ("kid", "kid_proj"):
                cells.append(fmt(df.attrs[c], 2))
            elif c not in df or df[c].notna().sum() < 0.5 * len(df):
                cells.append("--")
            else:
                cells.append(fmt(float(df[c].mean()), dec))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (OUT / fname).write_text("\n".join(lines) + "\n")
    print(fname)
    print("\n".join(lines))


RAW = [("raw_line_recall_t3", 2), ("raw_edge_prec_t3", 2), ("raw_sil_iou", 2), ("palette_de", 1), ("cut_residual", 1),
       ("clip_style_dedup", 3), ("kid", 2), ("palette_js", 2), ("chroma", 0), ("dino_gt", 2), ("nn_train_dino", 2)]
PLAN = [("proj_line_recall_t3", 2), ("proj_edge_prec_t3", 2), ("proj_sil_iou", 2), ("proj_decoration_density", 1),
        ("proj_cut_residual_fg", 1), ("proj_clip_style_dedup", 3), ("kid_proj", 2), ("proj_dino_gt", 2)]


def stats_table():
    f = ROOT / "outputs/stats_drawing_level.csv"
    df = pd.read_csv(f)
    names = {"cutcraft": "CutCraft", "blora": "B-LoRA", "instantstyle_guide": "InstantStyle + guide",
             "fulllora_guide": "Conv.\\ LoRA + guide", "cutline": "cut-line block"}
    mnames = {"raw_line_recall_t3": "line recall", "raw_sil_iou": "silhouette IoU", "clip_style_dedup": "CLIP style",
              "palette_js": "colour-mix distance", "proj_line_recall_t3": "plan line recall",
              "proj_clip_style_dedup": "plan CLIP style", "proj_dino_gt": "plan similarity to original",
              "proj_decoration_density": "plan decoration density", "proj_cut_residual_fg": "image-to-plan change",
              "proj_edge_prec_t3": "plan edge precision", "proj_purity": "plan purity", "proj_pieces_below_alpha": "pieces $<\\alpha$"}
    lines = []
    prev = None
    for _, r in df.iterrows():
        comp = f"{names.get(r.a, r.a)} vs {names.get(r.b, r.b)}"
        if comp != prev and prev is not None:
            lines.append(r"\midrule")
        prev = comp
        dec = 1 if abs(r.mean_a) > 5 else 3
        ph = fmt(r.p_holm, 3) if isinstance(r.p_holm, float) and np.isfinite(r.p_holm) else "(expl.)"
        if ph not in ("(expl.)",) and r.p_holm < 0.001:
            ph = "$<$0.001"
        lines.append(" & ".join([comp, mnames.get(r.metric, r.metric), str(int(r.n)), f"{r.mean_a:.{dec}f}", f"{r.mean_b:.{dec}f}",
                                 f"{r['diff']:+.{dec}f} [{r.ci_lo:+.{dec}f}, {r.ci_hi:+.{dec}f}]", f"{r.r_rb:+.2f}", ph]) + r" \\")
    (OUT / "table_stats.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def svg_table():
    names = [("blora", "B-LoRA"), ("instantstyle_guide", "InstantStyle + guide"), ("fulllora_guide", "Conv.\\ LoRA + guide"),
             ("cutline", "Cut-line block"), ("cutcraft", "CutCraft-SDXL"), ("qwen_q1", "CutCraft-Qwen")]
    frames = {}
    for f, lab in (("svg_validation.csv", "raw"), ("svg_validation_enforced.csv", "enforced")):
        fp = ROOT / "outputs" / f
        if fp.exists():
            frames[lab] = pd.read_csv(fp)
    lines = []
    for m, name in names:
        for lab, df in frames.items():
            d = df[df.method == m]
            if not len(d):
                continue
            rec = fmt(d.plan_line_recall.mean(), 2) if ("plan_line_recall" in d and lab == "enforced") else "--"
            lines.append(f"{name} & {lab} & {d.pieces.mean():.0f} & {(d.pieces_below_min > 0).mean():.2f} & "
                         f"{d.smallest_mm2.mean():.1f} & {d.pieces_with_thin_strip.mean():.1f} & {rec} \\\\")
    (OUT / "table_svg.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def supp_tables():
    methods = [("B-LoRA", "blora"), ("Prompt only + ControlNet", "prompt_cn"), ("InstantStyle + CraftGuide", "instantstyle_guide"),
               ("Conv.\\ LoRA + CraftGuide", "fulllora_guide"), ("Cut-line block", "cutline"), ("CutCraft", "cutcraft")]
    cols = []
    for t in (2, 3, 5):
        cols += [(f"raw_line_recall_t{t}", 2), (f"raw_edge_prec_t{t}", 2)]
    for t in (2, 3, 5):
        cols += [(f"proj_line_recall_t{t}", 2)]
    write([("Ku Shulan originals", "ksl_original"), ("midrule", None)] + methods, cols, "table_tol.tex")
    ab = pd.read_csv(ROOT / "outputs/rev_projection_ablation.csv")
    order = ["raw", "colour only", "ground only", "colour + merge (no ground)", "full projection"]
    lines = []
    for m, name in (("cutline", "Cut-line block"), ("cutcraft", "CutCraft")):
        d = ab[ab.method == m].copy()
        d["content"] = d.file.str.replace(r"_s_seed\d+\.png$", "", regex=True)
        g = d.groupby(["variant", "content"])[["line_recall_t3", "edge_prec_t3", "sil_iou", "decoration_density"]].mean().groupby("variant").mean()
        for v in order:
            r = g.loc[v]
            lines.append(f"{name} & {v} & {r.line_recall_t3:.3f} & {r.edge_prec_t3:.2f} & {r.sil_iou:.3f} & {r.decoration_density:.1f} \\\\")
        lines.append(r"\midrule")
    (OUT / "table_projabl.tex").write_text("\n".join(lines[:-1]) + "\n")
    rows = [("B-LoRA", "blora"), ("B-LoRA + CraftGuide", "blora_guide"), ("Prompt only + ControlNet", "prompt_cn"),
            ("StyleAligned + ControlNet", "stylealigned"),
           ("InstantStyle + ControlNet", "instantstyle"), ("InstantStyle + CraftGuide", "instantstyle_guide"),
            ("Conv.\\ LoRA + ControlNet", "fulllora"), ("Conv.\\ LoRA + CraftGuide", "fulllora_guide"),
            ("Collection block", "coll"), ("Cut-line block", "cutline"), ("CutCraft-SDXL", "cutcraft"),
            ("Qwen-Image-Edit zero-shot (Q0)", "qwen_q0"), ("Qwen + Canny-edge LoRA (Q1$'$)", "qwen_q1_canny"), ("CutCraft-Qwen (Q1)", "qwen_q1")]
    write([("Ku Shulan originals", "ksl_original"), ("midrule", None)] + rows, RAW, "table_indomain.tex", subset=["indomain"])
    write(rows, RAW, "table_ood.tex", subset=["repo", "ood"])



def robust_table():
    """Perturbed drawings (seed 0); structure scored against the original drawing, raw images only."""
    cols = [("raw_line_recall_t3", 2), ("raw_sil_iou", 2), ("palette_de", 1), ("clip_style_dedup", 3), ("palette_js", 2), ("chroma", 0)]
    rows = [("clean drawings", "cutcraft"), ("broken strokes (80 gaps)", "robust_broken"), ("thick strokes (9 px)", "robust_thick"),
            ("thin strokes (1 px)", "robust_thin"), ("jittered strokes", "robust_jitter")]
    lines = []
    for lab, m in rows:
        d = G / m
        if not (d / "rev_metrics.csv").exists():
            lines.append(lab + " & " + " & ".join([r"\pending{}"] * len(cols)) + r" \\")
            continue
        a = pd.read_csv(d / "metrics.csv").merge(pd.read_csv(d / "rev_metrics.csv").drop(columns=["content", "style", "seed", "set"], errors="ignore"), on="file")
        a = a[a.seed == 0]
        g = a.groupby("content")[[c for c, _ in cols]].mean()
        lines.append(lab + " & " + " & ".join(fmt(float(g[c].mean()), k) for c, k in cols) + r" \\")
    (OUT / "table_robust.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    svg_table()
    supp_tables()
    robust_table()
    write([("Ku Shulan originals", "ksl_original"), ("midrule", None),
           ("B-LoRA (repository workflow)", "blora"), ("B-LoRA + CraftGuide", "blora_guide"),
           ("Prompt only + ControlNet", "prompt_cn"), ("B-LoRA style block + ControlNet", "blora_style_cn"),
           ("StyleAligned + ControlNet", "stylealigned"),
           ("InstantStyle + ControlNet", "instantstyle"), ("InstantStyle + CraftGuide", "instantstyle_guide"),
           ("Conv.\\ LoRA + ControlNet", "fulllora"), ("Conv.\\ LoRA + CraftGuide", "fulllora_guide"),
           ("Conv.\\ LoRA + cut-line training", "fulllora_cl"), ("Conv.\\ LoRA + cut-line training + CraftGuide", "fulllora_cl_guide"),
           ("midrule", None),
           ("Collection block", "coll"), ("Collection block + CraftGuide", "coll_guide"),
           ("Cut-line block", "cutline"), ("CutCraft-SDXL (cut-line + CraftGuide)", "cutcraft"),
           ("Cut-line block, strict split", "cutline_strict"), ("Cut-line block, strict split + CraftGuide", "cutcraft_strict"),
           ("midrule", None),
           ("Qwen-Image-Edit-2511 zero-shot (Q0)", "qwen_q0"), ("Qwen + Canny-edge LoRA (Q1$'$)", "qwen_q1_canny"), ("CutCraft-Qwen (cut-line LoRA, Q1)", "qwen_q1"),
           ("CutCraft-Qwen + CraftGuide (Q2)", "qwen_q2")],
          RAW, "table_main.tex")
    write([("Ku Shulan originals", "ksl_original"), ("midrule", None),
           ("B-LoRA", "blora"), ("InstantStyle + CraftGuide", "instantstyle_guide"),
           ("Conv.\\ LoRA + CraftGuide", "fulllora_guide"),
           ("Cut-line block", "cutline"), ("CutCraft-SDXL", "cutcraft"),
           ("Qwen-Image-Edit zero-shot", "qwen_q0"), ("CutCraft-Qwen", "qwen_q1")], PLAN, "table_plan.tex")
    if (ROOT / "outputs/stats_drawing_level.csv").exists():
        stats_table()
