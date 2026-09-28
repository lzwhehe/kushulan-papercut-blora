"""Robustness inputs (review #11): perturbed versions of the 34 evaluation drawings.

broken : 80 random gaps (disk radius 12 px at 1024) punched into the strokes
thick  : strokes dilated to about 9 px
thin   : 1-px skeleton
jitter : smooth random displacement field (amplitude 6 px, sigma 40 px) plus 15 small gaps
Structure metrics are always computed against the original, unperturbed drawing.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/contents"
OUT = ROOT / "outputs/contents_robust"


def gaps(ink, n, r, rng):
    ys, xs = np.nonzero(ink)
    out = ink.copy()
    yy, xx = np.ogrid[:ink.shape[0], :ink.shape[1]]
    for i in rng.choice(len(ys), min(n, len(ys)), replace=False):
        out[(yy - ys[i]) ** 2 + (xx - xs[i]) ** 2 <= r * r] = False
    return out


def jitter(ink, amp, sigma, rng):
    h, w = ink.shape
    dy = ndi.gaussian_filter(rng.normal(size=(h, w)), sigma)
    dx = ndi.gaussian_filter(rng.normal(size=(h, w)), sigma)
    dy *= amp / (np.abs(dy).max() + 1e-8)
    dx *= amp / (np.abs(dx).max() + 1e-8)
    yy, xx = np.mgrid[:h, :w]
    return ndi.map_coordinates(ink.astype(float), [yy + dy, xx + dx], order=1) > 0.5


def main():
    contents = json.loads((SRC / "contents.json").read_text())
    for kind in ("broken", "thick", "thin", "jitter"):
        (OUT / kind).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(contents):
            rng = np.random.default_rng(1000 + i)
            ink = np.asarray(Image.open(SRC / f"{c['id']}.png").convert("L")) < 128
            if kind == "broken":
                x = gaps(ink, 80, 12, rng)
            elif kind == "thick":
                x = ndi.binary_dilation(ink, morphology.disk(3))
            elif kind == "thin":
                x = morphology.skeletonize(ink)
            else:
                x = gaps(jitter(ink, 6, 40, rng), 15, 6, rng)
            Image.fromarray(np.where(x, 0, 255).astype(np.uint8)).save(OUT / kind / f"{c['id']}.png")
        (OUT / kind / "contents.json").write_text(json.dumps(contents, ensure_ascii=False, indent=1))
    print("done")


if __name__ == "__main__":
    main()
