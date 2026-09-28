"""Palette-proportion divergence: Jensen-Shannon distance between an output's paper-colour
shares (from its cut projection) and those of the closest Ku Shulan work in the corpus. Added after visual inspection
showed that palette distance alone cannot distinguish Ku Shulan's colour proportions
from outputs dominated by the palette's rare greys."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")
K = len(PAL.lab)


def shares(labels):
    fg = labels[labels < K]
    if fg.size == 0:
        return None
    return np.bincount(fg.ravel(), minlength=K) / fg.size


def _corpus_one(p):
    rgb = priors.load_rgb(ROOT / p, 512)
    return shares(priors.cut_project(rgb, PAL)[1])


def corpus_hists():
    cache = ROOT / "outputs/data/corpus_palette_shares.npy"
    if cache.exists():
        return np.load(cache)
    from concurrent.futures import ProcessPoolExecutor
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]] + split["held_out"]
    with ProcessPoolExecutor(24) as ex:
        H = [h for h in ex.map(_corpus_one, paths) if h is not None]
    H = np.stack(H)
    np.save(cache, H)
    return H


def nn_js(s, H):
    return float(min(jensenshannon(s, h, base=2) for h in H))


def score_dir(m, H):
    d = G / m
    rows = []
    for f in sorted(d.glob("*.labels.npy")):
        s = shares(np.load(f))
        rows.append({"file": f.name.replace(".labels.npy", ".png"),
                     "palette_js": np.nan if s is None else nn_js(s, H)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    H = corpus_hists()
    for m in sys.argv[1:]:
        df = score_dir(m, H)
        for name in ("metrics.csv", "metrics_projected.csv"):
            p = G / m / name
            if p.exists():
                x = pd.read_csv(p).drop(columns=["palette_js"], errors="ignore").merge(df, on="file", how="left")
                x.to_csv(p, index=False)
        raw = pd.read_csv(G / m / "metrics.csv")
        pj = pd.read_csv(G / m / "metrics_projected.csv") if (G / m / "metrics_projected.csv").exists() else raw
        print(f"{m:18s} nnJS={raw.palette_js.mean():.3f} chroma raw={raw.chroma.mean():.1f} proj={pj.chroma.mean():.1f} "
              f"| nnJS in-domain={raw[raw.set=='indomain'].palette_js.mean():.3f} ood+repo={raw[raw.set!='indomain'].palette_js.mean():.3f}")
