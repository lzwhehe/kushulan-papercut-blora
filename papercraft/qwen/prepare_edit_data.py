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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "outputs/data/split.json"))
    ap.add_argument("--maps", default="dense,coarse", help="dense,coarse | canny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_flip", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(a.split).read_text())["train"]
    meta = []
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
                inp = np.where(lines, 0, 255).astype(np.uint8)  # black lines on white
                iname = f"inputs/{n:04d}_{kind}{'_f' if flip else ''}.png"
                Image.fromarray(np.ascontiguousarray(inp)).convert("RGB").save(out / iname)
                meta.append({"image": tname, "edit_image": iname, "prompt": INSTRUCTION.format(subject=subject),
                             "source": it["path"], "map": kind, "flip": flip})
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(len(items), "images ->", len(meta), "pairs in", out)


if __name__ == "__main__":
    main()
