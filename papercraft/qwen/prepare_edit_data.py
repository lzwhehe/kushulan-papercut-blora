"""Edit pairs for the Qwen-Image-Edit-2511 backbone (same data as the SDXL cut-line block).

For every training image of a split: input (edit_image) = automatically derived cut-line map, drawn as
black lines on white like a user's line drawing; target (image) = the catalogue image on white, 1024 px.
Dense and coarse maps give two pairs per image (the SDXL block drew one of them at random per step), and
horizontally flipped copies mirror the SDXL flip augmentation. --maps canny builds the Canny-edge control
dataset (Q1'). Writes <out>/{inputs,targets}/*.png and <out>/metadata.json for DiffSynth-Studio.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import priors  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = ("Turn this line drawing into a Ku Shulan style colour paste-up papercut of {subject}: "
               "flat pieces of saturated coloured paper with dots, crescents and saw-tooth motifs, "
               "no shading, on plain white paper, following the drawn lines.")


def fragment(lines, rng, drop=0.25, gaps=60, gap_r=10):
    """Open-stroke augmentation: split the cut-line skeleton at junctions, drop a share of the segments
    and punch gaps, so that many strokes end in free space. Every remaining fragment still lies on a
    colour boundary of the target, so the model learns to keep open strokes instead of ignoring them."""
    from scipy import ndimage as ndi
    from skimage import morphology
    sk = morphology.skeletonize(lines)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    seg, n = ndi.label(sk & ~(sk & (nb >= 3)), structure=np.ones((3, 3)))
    keep = np.r_[False, rng.random(n) >= drop]
    kept = keep[seg]
    ys, xs = np.nonzero(kept)
    yy, xx = np.ogrid[:lines.shape[0], :lines.shape[1]]
    for i in rng.choice(len(ys), min(gaps, len(ys)), replace=False) if len(ys) else []:
        kept[(yy - ys[i]) ** 2 + (xx - xs[i]) ** 2 <= gap_r * gap_r] = False
    return lines & ndi.binary_dilation(kept, iterations=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "outputs/data/split.json"))
    ap.add_argument("--maps", default="dense,coarse", help="dense,coarse | canny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fragment", type=float, default=0.0, help="share of inputs given open-stroke augmentation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(a.split).read_text())["train"]
    meta = []
    rng = np.random.default_rng(a.seed)
    for n, it in enumerate(items):
        stem = str(Path(it["path"])).replace("/", "__")
        tgt = (priors.load_rgb(ROOT / it["path"], 1024) * 255).round().astype(np.uint8)
        subject = it["caption"].replace("a animal", "an animal")
        for flip in ([False] if a.no_flip else [False, True]):
            t = tgt[:, ::-1] if flip else tgt
            tname = f"targets/{n:04d}{'_f' if flip else ''}.png"
            Image.fromarray(np.ascontiguousarray(t)).save(out / tname)
            for kind in a.maps.split(","):
                lines = np.asarray(Image.open(ROOT / f"outputs/cutlines/{stem}__{kind}.png").convert("L")) < 128
                if flip:
                    lines = lines[:, ::-1]
                frag = a.fragment > 0 and rng.random() < a.fragment
                if frag:
                    lines = fragment(np.ascontiguousarray(lines), rng)
                inp = np.where(lines, 0, 255).astype(np.uint8)  # black lines on white
                iname = f"inputs/{n:04d}_{kind}{'_f' if flip else ''}.png"
                Image.fromarray(np.ascontiguousarray(inp)).convert("RGB").save(out / iname)
                meta.append({"image": tname, "edit_image": iname, "prompt": INSTRUCTION.format(subject=subject),
                             "source": it["path"], "map": kind, "flip": flip, "fragmented": bool(frag)})
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(len(items), "images ->", len(meta), "pairs in", out)


if __name__ == "__main__":
    main()
