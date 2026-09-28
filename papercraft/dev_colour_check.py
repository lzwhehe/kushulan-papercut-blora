"""Review #22: replicate the post-hoc colour measures on the independent development set."""
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

import palette_usage as PU
import priors

ROOT = Path(__file__).resolve().parents[1]
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")
H = PU.corpus_hists()
K = len(PAL.lab)
DIRS = {"prompt only": "dev/prompt_cn", "collection block": "dev/cn0.6", "collection block + guide": "dev3/coll_g0.6",
        "cut-line block": "dev3/cutline_cn0.6", "cut-line block + guide (CutCraft)": "dev3/cutline_g1.0"}

for name, d in DIRS.items():
    ch, js = [], []
    for f in sorted((ROOT / "outputs" / d).glob("*_seed*.png")):
        rgb = priors.load_rgb(f, 512)
        lab = priors.to_lab(rgb)
        fg = priors.foreground_mask(lab)
        ch.append(float(np.hypot(lab[..., 1], lab[..., 2])[fg].mean()))
        s = PU.shares(priors.cut_project(rgb, PAL)[1])
        if s is not None:
            js.append(PU.nn_js(s, H))
    print(f"{name:36s} n={len(ch)} chroma={np.mean(ch):.1f} colour-mix={np.mean(js):.3f}")
