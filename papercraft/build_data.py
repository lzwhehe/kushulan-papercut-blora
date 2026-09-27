"""Build the train / test split, the Ku Shulan palette and corpus craft statistics.

Outputs (under ``--out``):
  split.json          training style images, held-out test triplets, style references
  palette.json        CIELAB palette fitted on training images only
  corpus_stats.csv    craft metrics for every catalogued image
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import priors

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
PAT = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Pattern_symbols"
PAIRS = ROOT / "data/涂方圆/库淑兰-对照图"
CAT_EN = {"人物": "person", "动物": "animal", "日常": "daily object", "植物": "plant",
          "窗花": "window flower", "边框": "border"}
REP_NOUN = {"animals": "animal", "daily": "daily object", "frame": "decorative border",
            "person": "folk figure", "plants": "plant", "window decoration": "round window flower"}


def dino_embed(paths, model, proc, dev):
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.fromarray((priors.load_rgb(p, 256) * 255).astype(np.uint8)) for p in paths[i:i + 16]]
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt").to(dev)
            f = model(**x).pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/data"))
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rep = sorted(glob.glob(str(REP / "*/*")))
    pat = sorted(glob.glob(str(PAT / "*/*")))
    # outline triplets: <cat>-<id>-轮廓(线).*, <cat>-<id>-处理后.jpg
    triplets = []
    for f in sorted(PAIRS.glob("*-处理后.jpg")):
        key = f.name.replace("-处理后.jpg", "")
        outl = sorted(PAIRS.glob(key + "-轮廓*"))
        if outl:
            cat = key.split("-")[0]
            triplets.append({"id": key, "category": CAT_EN[cat], "outline": str(outl[0].relative_to(ROOT)),
                             "reference": str(f.relative_to(ROOT))})

    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    e_cat = dino_embed(rep + pat, model, proc, dev)
    e_ref = dino_embed([str(ROOT / t["reference"]) for t in triplets], model, proc, dev)
    sim = e_ref @ e_cat.T
    held = set()
    for t, s in zip(triplets, sim):
        # hold out every catalogue image that is a near duplicate of the test work
        idx = torch.nonzero(s > 0.93).flatten().tolist() or [int(s.argmax())]
        t["matched_catalogue"] = [str(Path((rep + pat)[i]).relative_to(ROOT)) for i in idx]
        t["match_sim"] = [round(float(s[i]), 3) for i in idx]
        held.update(idx)
    train = [p for i, p in enumerate(rep + pat) if i not in held]

    def caption(p):
        p = Path(p)
        if str(PAT) in str(p):
            name = re.sub(r"(?<!^)(?=[A-Z])", " ", p.parent.name).lower()
            return f"a {name} motif"
        return f"a {REP_NOUN[p.parent.name]}"

    # four single-image style references for the per-reference B-LoRA baseline,
    # chosen from training images of different categories (fixed seed)
    rng = np.random.default_rng(7)
    style_refs = []
    for sub in ["animals", "person", "plants", "window decoration"]:
        cands = [p for p in train if f"/{sub}/" in p]
        style_refs.append(str(Path(cands[rng.integers(len(cands))]).relative_to(ROOT)))

    split = {
        "train": [{"path": str(Path(p).relative_to(ROOT)), "caption": caption(p)} for p in train],
        "held_out": sorted(str(Path((rep + pat)[i]).relative_to(ROOT)) for i in held),
        "test_indomain": triplets,
        "style_refs": style_refs,
    }
    (out / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1))
    print(f"train={len(train)} held_out={len(held)} test_triplets={len(triplets)}")

    pal = priors.fit_palette(train, k=args.k)
    pal.save(out / "palette.json")
    print("palette", json.loads((out / "palette.json").read_text())["hex"])

    rows = []
    for p in rep + pat:
        m = priors.craft_metrics(priors.load_rgb(p, 512), pal)
        m.update(path=str(Path(p).relative_to(ROOT)), subset="pattern" if p in pat else "representative",
                 symmetry=priors.symmetry_score(priors.load_rgb(p, 256)))
        rows.append(m)
    pd.DataFrame(rows).to_csv(out / "corpus_stats.csv", index=False)


if __name__ == "__main__":
    main()
