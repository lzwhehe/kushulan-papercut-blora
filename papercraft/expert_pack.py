"""Blinded stimulus pack for the expert study (review #02, Supplementary S6).

20 drawings drawn at random (seed 0) from the 34 held-out drawings; for each, the plan (cut projection,
seed 0 / animal reference) of B-LoRA, InstantStyle, SDXL LoRA + ControlNet, the untrained editing model and our method.
Stimuli are shown in random order under neutral codes, one per page with the line drawing beside it,
and Ku Shulan originals of the in-domain drawings are added as anchors. Writes stimuli.pdf, a rating
sheet template (ratings_template.csv) and the key (key_UNBLIND.csv, to be withheld from raters).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
OUT = ROOT / "outputs/expert_pack"
METHODS = {"blora": "{cid}_s0_seed0", "instantstyle": "{cid}_s0_seed0", "fulllora": "{cid}_s_seed0",
           "qwen_q0": "{cid}_s_seed0", "qwen_q1": "{cid}_s_seed0"}
SCALES = ["style resemblance (1-5)", "faithfulness to drawing (1-5)", "motif/colour appropriateness (1-5)",
          "feasibility to cut and paste (1-5)", "overall quality (1-5)", "comments"]


def plan_image(m, cid, pal_rgb):
    stem = METHODS[m].format(cid=cid)
    lab = G / m / f"{stem}.labels.npy"
    if not lab.exists():  # file naming differs between sampling scripts
        cands = sorted((G / m).glob(f"{cid}_s*seed0.labels.npy"))
        lab = cands[0]
    return pal_rgb[np.load(lab)]


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    pal_rgb = np.clip(priors.color.lab2rgb(pal.lab_with_white[None])[0], 0, 1)
    contents = json.loads((ROOT / "outputs/contents/contents.json").read_text())
    rng = np.random.default_rng(0)
    chosen = [contents[i] for i in sorted(rng.choice(len(contents), 20, replace=False))]
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    refs = {f"ind_{t['id']}": t["reference"] for t in split["test_indomain"]}
    items = []
    for c in chosen:
        for m in METHODS:
            items.append({"drawing": c["id"], "method": m})
        if c["id"] in refs:
            items.append({"drawing": c["id"], "method": "original"})
    order = rng.permutation(len(items))
    OUT.mkdir(parents=True, exist_ok=True)
    key = []
    with PdfPages(OUT / "stimuli.pdf") as pdf:
        for n, i in enumerate(order, 1):
            it = items[i]
            code = f"S{n:03d}"
            if it["method"] == "original":
                img = np.asarray(ImageOps.pad(Image.open(ROOT / refs[it["drawing"]]).convert("RGB"), (512, 512), color="white")) / 255  # keep aspect ratio
            else:
                img = plan_image(it["method"], it["drawing"], pal_rgb)
            drw = np.asarray(Image.open(ROOT / f"outputs/contents/{it['drawing']}.png").convert("RGB").resize((512, 512))) / 255
            fig, ax = plt.subplots(1, 2, figsize=(11.69, 6.2))
            ax[0].imshow(drw); ax[0].set_title("line drawing", fontsize=10)
            ax[1].imshow(img); ax[1].set_title(f"design {code}", fontsize=10)
            for a in ax:
                a.axis("off")
            pdf.savefig(fig); plt.close(fig)
            key.append({"code": code, **it})
    pd.DataFrame(key).to_csv(OUT / "key_UNBLIND.csv", index=False)
    pd.DataFrame({"code": [k["code"] for k in key], **{s: "" for s in SCALES}}).to_csv(OUT / "ratings_template.csv", index=False)
    print(len(key), "stimuli;", pd.DataFrame(key).method.value_counts().to_dict())


if __name__ == "__main__":
    main()
