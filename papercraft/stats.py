"""Drawing-level statistics for the revision (review #03, #09, #10).

Unit of analysis = line drawing. Repeated outputs of a method for the same drawing (two seeds, or
two style references for reference-based methods) are averaged first, so every comparison has
n = 34 drawings (n = 19 for measures that need Ku Shulan's original). For each comparison we report
the mean of each method, the paired mean difference with a 95 % bootstrap CI (10,000 resamples of
drawings), the Wilcoxon signed-rank p on per-drawing means and the matched-pairs rank-biserial
correlation. A small primary family is Holm-corrected; everything else is labelled exploratory.
The primary family was fixed during the revision, after the first round of results had been seen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"


def per_drawing(method):
    d = G / method
    a = pd.read_csv(d / "metrics.csv")
    r = pd.read_csv(d / "rev_metrics.csv")
    p = pd.read_csv(d / "metrics_projected.csv")[["file", "clip_style", "dino_gt", "edge_f1"]].rename(
        columns={"clip_style": "proj_clip_style", "dino_gt": "proj_dino_gt", "edge_f1": "proj_edge_f1_old"})
    df = a.merge(r.drop(columns=[c for c in ("content", "style", "seed", "set") if c in r], errors="ignore"), on="file")
    df = df.merge(p, on="file", how="left")
    num = df.select_dtypes("number").columns.drop(["seed"], errors="ignore")
    return df.groupby("content")[num].mean()


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(a, b, metric, higher_better=True, n_boot=10000, seed=0):
    A, B = per_drawing(a), per_drawing(b)
    j = A[[metric]].join(B[[metric]], lsuffix="_a", rsuffix="_b").dropna()
    x, y = j[f"{metric}_a"].values, j[f"{metric}_b"].values
    d = x - y
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = wilcoxon(x, y).pvalue if not np.allclose(d, 0) else 1.0
    return {"a": a, "b": b, "metric": metric, "n": len(d), "mean_a": x.mean(), "mean_b": y.mean(),
            "diff": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "p": p, "r_rb": rank_biserial(d), "better": "a" if (d.mean() > 0) == higher_better else "b"}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


PRIMARY_METRICS = [("raw_line_recall_t3", True), ("raw_sil_iou", True), ("clip_style_dedup", True), ("palette_js", False)]


def main():
    comps = [("cutcraft", "blora"), ("cutcraft", "fulllora_guide"), ("cutcraft", "instantstyle_guide")]
    rows = []
    for a, b in comps:
        if not (G / b / "rev_metrics.csv").exists():
            print("skip (not scored yet):", b)
            continue
        for m, hb in PRIMARY_METRICS:
            r = compare(a, b, m, hb)
            r["family"] = "primary"
            rows.append(r)
    # CraftGuide increment on the final, projected output (review #03)
    for m, hb in [("proj_line_recall_t3", True), ("proj_edge_prec_t3", True), ("proj_decoration_density", True),
                  ("proj_clip_style_dedup", True), ("proj_dino_gt", True), ("proj_purity", True),
                  ("proj_pieces_below_alpha", False), ("proj_cut_residual_fg", False)]:
        r = compare("cutcraft", "cutline", m, hb)
        r["family"] = "primary" if m in ("proj_line_recall_t3", "proj_clip_style_dedup") else "exploratory"
        rows.append(r)
    df = pd.DataFrame(rows)
    prim = df.family == "primary"
    df.loc[prim, "p_holm"] = holm(df.loc[prim, "p"].values)
    df.to_csv(ROOT / "outputs/stats_drawing_level.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df[["family", "a", "b", "metric", "n", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p", "p_holm", "r_rb"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
