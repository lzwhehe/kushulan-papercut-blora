"""Craft statistics of the Ku Shulan corpus (CPU only)."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import priors

ROOT = Path(__file__).resolve().parents[1]


def one(p):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    rgb = priors.load_rgb(ROOT / p, 512)
    m = priors.craft_metrics(rgb, pal)
    m["symmetry"] = priors.symmetry_score(priors.load_rgb(ROOT / p, 256))
    m["path"] = p
    parts = Path(p).parts
    m["subset"] = "pattern" if "Pattern_symbols" in p else "representative"
    m["category"] = parts[-2]
    return m


if __name__ == "__main__":
    s = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in s["train"]] + s["held_out"]
    with ProcessPoolExecutor(24) as ex:
        rows = list(ex.map(one, paths))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/data/corpus_stats.csv", index=False)
    print(df.groupby("subset")[["purity", "palette_de", "cut_residual", "fragments", "chroma", "symmetry"]].describe().T.round(3).to_string())
