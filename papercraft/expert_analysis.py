"""Analyse the returned expert questionnaires (expert_survey.py).

Usage: python expert_analysis.py <folder with kushulan_expert_*.json> [--out results/cutcraft/expert_study]

Unblinds the ratings with key_UNBLIND.csv and outputs/expert_survey/repeats.csv and writes:
  ratings_long.csv   one row per rater x stimulus x dimension (repeats excluded)
  method_means.csv   per method and dimension: mean over raters of the rater's mean, SD, n raters
  paired_tests.csv   ours against every other method: drawing-level means (averaged over raters),
                     mean difference, 95% bootstrap CI, Wilcoxon signed-rank p, Holm-adjusted per dimension
  reliability.csv    ICC(2,1) and ICC(2,k) per dimension over the main stimuli; within-rater
                     consistency on the repeated stimuli (mean absolute difference, exact agreement)
  raters.csv         background of the raters (as entered; no names are collected)
  report.md          a short summary of the above
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "results/cutcraft/study_materials/expert_pack/key_UNBLIND.csv"
REPEATS = ROOT / "outputs/expert_survey/repeats.csv"
DIMS = {"style": "style resemblance", "faith": "faithfulness to drawing", "motif": "motif/colour appropriateness",
        "cut": "feasibility to cut and paste", "overall": "overall quality"}
NAMES = {"qwen_q1": "ours", "qwen_q0": "untrained editing model", "fulllora": "SDXL LoRA + ControlNet",
         "instantstyle": "InstantStyle", "blora": "B-LoRA", "original": "reference redrawings (anchors)"}


def load(folder):
    rows, raters = [], []
    for f in sorted(Path(folder).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf8"))
        rid = d["rater"]["id"]
        raters.append({**d["rater"], "started": d.get("started"), "finished": d.get("finished"), "file": f.name})
        for r in d["ratings"]:
            for k in DIMS:
                if r.get(k) is not None:
                    rows.append({"rater": rid, "code": r["code"], "dim": k, "score": int(r[k]), "sec": r.get("sec")})
    return pd.DataFrame(rows), pd.DataFrame(raters)


def icc2(m):
    """ICC(2,1) and ICC(2,k) for a targets x raters matrix without missing values (Shrout & Fleiss)."""
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    single = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    average = (msr - mse) / (msr + (msc - mse) / n)
    return single, average


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def md(df, index=True):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(ROOT / "results/cutcraft/expert_study"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    long, raters = load(a.folder)
    if long.empty:
        raise SystemExit("no ratings found")
    key = pd.read_csv(KEY)
    rep = pd.read_csv(REPEATS) if REPEATS.exists() else pd.DataFrame(columns=["code", "repeats"])
    main_ = long[~long.code.isin(rep.code)].merge(key, on="code")
    main_["method"] = main_.method.map(lambda m: NAMES.get(m, m))
    main_.to_csv(out / "ratings_long.csv", index=False)
    raters.to_csv(out / "raters.csv", index=False)

    # per method: rater means, then mean and SD over raters
    rm = main_.groupby(["dim", "method", "rater"]).score.mean().reset_index()
    mm = rm.groupby(["dim", "method"]).score.agg(["mean", "std", "count"]).reset_index().rename(columns={"count": "raters"})
    mm.to_csv(out / "method_means.csv", index=False)

    # ours vs each other method, paired over drawings (scores averaged over raters)
    dm = main_.groupby(["dim", "method", "drawing"]).score.mean().reset_index()
    rng = np.random.default_rng(0)
    tests = []
    for dim in DIMS:
        sub = dm[dm.dim == dim].pivot(index="drawing", columns="method", values="score")
        for other in [c for c in sub.columns if c != "ours"]:
            pair = sub[["ours", other]].dropna()
            diff = (pair["ours"] - pair[other]).to_numpy()
            if len(diff) < 2:
                continue
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
            p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            tests.append({"dim": dim, "other": other, "n_drawings": len(diff), "diff": diff.mean(),
                          "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5), "p": p})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["p_holm"] = tests.groupby("dim").p.transform(lambda s: holm(s.to_numpy()))
    tests.to_csv(out / "paired_tests.csv", index=False)

    # reliability
    rel = []
    for dim in DIMS:
        mat = main_[main_.dim == dim].pivot_table(index="code", columns="rater", values="score").dropna()
        if mat.shape[1] >= 2 and mat.shape[0] >= 2:
            s1, sk = icc2(mat.to_numpy(float))
            rel.append({"dim": dim, "measure": "ICC(2,1)", "value": s1, "n_stimuli": mat.shape[0], "n_raters": mat.shape[1]})
            rel.append({"dim": dim, "measure": "ICC(2,k)", "value": sk, "n_stimuli": mat.shape[0], "n_raters": mat.shape[1]})
        if not rep.empty:
            r2 = long[long.code.isin(rep.code) & (long.dim == dim)].merge(rep, on="code")
            r1 = long[(long.dim == dim)][["rater", "code", "score"]].rename(columns={"code": "repeats", "score": "first"})
            both = r2.merge(r1, on=["rater", "repeats"])
            if len(both):
                rel.append({"dim": dim, "measure": "repeat mean abs. difference", "value": (both.score - both["first"]).abs().mean(),
                            "n_stimuli": len(both), "n_raters": both.rater.nunique()})
                rel.append({"dim": dim, "measure": "repeat exact agreement", "value": (both.score == both["first"]).mean(),
                            "n_stimuli": len(both), "n_raters": both.rater.nunique()})
    rel = pd.DataFrame(rel)
    rel.to_csv(out / "reliability.csv", index=False)

    # report
    L = [f"# Expert study: {raters.shape[0]} raters, {main_.code.nunique()} stimuli", ""]
    L += ["## Mean scores (1-5; mean over raters of each rater's mean)", "",
          md(mm.pivot(index="method", columns="dim", values="mean")[list(DIMS)].round(2)), ""]
    if not tests.empty:
        L += ["## Ours minus each method (drawing-level, Holm within dimension)", "",
              md(tests.assign(**{"diff [95% CI]": tests.apply(lambda r: f"{r['diff']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", axis=1)})
              [["dim", "other", "n_drawings", "diff [95% CI]", "p_holm"]].round(4), index=False), ""]
    if not rel.empty:
        L += ["## Reliability", "", md(rel.round(3), index=False), ""]
    L += ["## Raters", "", md(raters.drop(columns=["file"]), index=False), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
