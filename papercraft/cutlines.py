"""Cut-line maps: the line drawing implied by a papercut.

A papercut's line drawing is the set of boundaries between its paper pieces. We
recover it by projecting the work onto the Ku Shulan palette and tracing label
boundaries. ``dense`` keeps every piece; ``coarse`` first merges pieces smaller
than 0.5 % of the canvas, leaving only the outline and main regions, which is
what a user's sketch typically contains.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cutlines"


def boundaries(labels):
    e = np.zeros_like(labels, bool)
    e[1:] |= labels[1:] != labels[:-1]
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return e


def cutline_maps(path, pal, res=1024):
    rgb = priors.load_rgb(path, 512)
    out = {}
    for name, frac in (("dense", 2e-4), ("coarse", 5e-3)):
        _, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=frac, smooth=2 if name == "dense" else 3)
        e = morphology.skeletonize(boundaries(labels))
        img = Image.fromarray(np.where(e, 0, 255).astype(np.uint8)).resize((res, res), Image.NEAREST)
        lines = np.asarray(img) < 128
        lines = ndi.binary_dilation(morphology.skeletonize(lines), morphology.disk(1))
        out[name] = Image.fromarray(np.where(lines, 0, 255).astype(np.uint8))
    return out


def _one(p):
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    maps = cutline_maps(ROOT / p, pal)
    stem = p.replace("/", "__")
    for k, im in maps.items():
        im.save(OUT / f"{stem}__{k}.png")
    return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]]
    with ProcessPoolExecutor(24) as ex:
        list(ex.map(_one, paths))
    print(len(paths), "works ->", len(list(OUT.glob("*.png"))), "maps")
