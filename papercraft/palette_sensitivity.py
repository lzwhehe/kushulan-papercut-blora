"""Review #24: palette size K. Palettes are fitted on the 257 training works only and evaluated on
the 25 held-out catalogue works: mean CIEDE2000 of foreground pixels to the nearest palette colour,
and the cut residual (mean change when the work is projected with that palette, alpha = 2e-4)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import priors

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    train = [ROOT / it["path"] for it in split["train"]]
    held = [ROOT / p for p in split["held_out"]]
    rows = []
    for k in (8, 12, 16, 24):
        pal = priors.fit_palette(train, k=k)
        de, res = [], []
        for p in held:
            rgb = priors.load_rgb(p, 384)
            lab = priors.to_lab(rgb)
            fg = priors.foreground_mask(lab)
            _, d = priors.nearest_palette(lab, pal.lab_with_white)
            de.append(d[fg].mean())
            proj = priors.cut_project(rgb, pal)[0]
            res.append(priors.color.deltaE_ciede2000(lab, priors.to_lab(proj))[fg].mean())
        rows.append({"K": k, "heldout_palette_de": float(np.mean(de)), "heldout_cut_residual": float(np.mean(res)),
                     "smallest_weight": float(pal.weight.min())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "outputs/palette_sensitivity.csv", index=False)
