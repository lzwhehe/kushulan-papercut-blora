"""Aggregate per-image metrics into tables with bootstrap CIs and paired tests.

python aggregate.py --methods name=dir[:projected] ... --baseline name --out table.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METRICS = [("palette_de", -1), ("purity", 1), ("cut_residual", -1), ("fragments", -1),
           ("edge_f1", 1), ("edge_r", 1), ("sil_iou", 1), ("clip_style", 1), ("clip_text", 1),
           ("dino_styleref", -1), ("dino_gt", 1)]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load(spec):
    name, d = spec.split("=", 1)
    proj = d.endswith(":projected")
    d = Path(d.replace(":projected", ""))
    df = pd.read_csv(d / ("metrics_projected.csv" if proj else "metrics.csv"))
    kid = json.loads((d / ("metrics_projected_kid.json" if proj else "metrics_kid.json")).read_text())
    df["method"] = name
    # average over style references so every method has one row per (content, seed)
    num = [m for m, _ in METRICS if m in df]
    agg = df.groupby(["content", "set", "seed"], as_index=False)[num].mean()
    agg["method"] = name
    return agg, kid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--subset", default=None, help="indomain | repo | ood | ood+repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames, kids = {}, {}
    for s in a.methods:
        df, k = load(s)
        if a.subset:
            df = df[df["set"].isin(a.subset.split("+"))]
        frames[df["method"].iloc[0]] = df
        kids[df["method"].iloc[0]] = k
    rows = []
    base = frames.get(a.baseline)
    for name, df in frames.items():
        r = {"method": name, "n": len(df), "clip_kid_x1000": kids[name]["clip_kid_x1000"]}
        for m, sign in METRICS:
            if m not in df or df[m].isna().all():
                continue
            mu, lo, hi = boot_ci(df[m])
            r[m] = mu
            r[m + "_ci"] = f"[{lo:.3g}, {hi:.3g}]"
            if base is not None and name != a.baseline and m in base:
                j = df.merge(base, on=["content", "seed"], suffixes=("", "_b"))[[m, m + "_b"]].dropna()
                if len(j) > 5 and not np.allclose(j[m], j[m + "_b"]):
                    r[m + "_p"] = wilcoxon(j[m], j[m + "_b"]).pvalue
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    show = ["method", "n"] + [m for m, _ in METRICS if m in out] + ["clip_kid_x1000"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
