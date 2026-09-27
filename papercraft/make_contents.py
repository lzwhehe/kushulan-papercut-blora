"""Build the evaluation content set as normalised binary line art (black on white, 1024 px).

  indomain/*  : 18 Ku Shulan outlines (涂方圆 triplets) with coloured ground truth
  repo/*      : 3 line arts used in the repository's 7.5 experiment
  ood/*       : 12 out-of-domain line arts sampled from SDXL, subjects disjoint from training
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/contents"
OOD = ["a rooster", "a rabbit", "a tiger", "a carp fish", "a crane bird", "a frog",
       "a cat", "a deer", "a teapot", "a bicycle", "a woman beating a drum", "a potted chrysanthemum"]
REPO = {"repo_fish": ("docs/assets/results/fish-content.png", "a fish"),
        "repo_bird": ("docs/assets/results/bird-content.png", "a bird with spread wings"),
        "repo_motif": ("docs/assets/results/motif-content.png", "a symmetric decorative motif")}
IND_NOUN = {"person": "a folk figure", "animal": "an animal", "daily object": "a daily object",
            "plant": "a plant", "window flower": "a round window flower", "border": "a decorative border"}


def normalise_lineart(rgb, res=1024, target_width=3):
    """Binary black-on-white lines with roughly constant stroke width."""
    g = rgb.mean(-1)
    t = min(filters.threshold_otsu(g), 0.85)
    lines = g < t
    lines = morphology.remove_small_objects(lines, 16)
    sk = morphology.skeletonize(lines)
    lines = ndi.binary_dilation(sk, morphology.disk(target_width // 2 + 0))
    img = np.where(lines, 0, 255).astype(np.uint8)
    return Image.fromarray(img).resize((res, res), Image.NEAREST)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    meta = []
    for t in split["test_indomain"]:
        cid = "ind_" + t["id"]
        rgb = priors.load_rgb(ROOT / t["outline"], 1024)
        normalise_lineart(rgb).save(OUT / f"{cid}.png")
        meta.append({"id": cid, "set": "indomain", "subject": IND_NOUN[t["category"]],
                     "category": t["category"], "reference": t["reference"]})
    for cid, (p, subj) in REPO.items():
        normalise_lineart(priors.load_rgb(ROOT / p, 1024)).save(OUT / f"{cid}.png")
        meta.append({"id": cid, "set": "repo", "subject": subj})

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for i, s in enumerate(OOD):
        # 4 seeds; keep the candidate with the least ink near the border (isolated subject)
        best = None
        for j in range(4):
            seed = 100 + 10 * i + j
            g = torch.Generator("cuda").manual_seed(seed)
            im = pipe(f"simple minimal black line drawing of {s}, single isolated subject, centered, "
                      f"coloring book outline, bold clean black outlines, lots of empty white space around it",
                      negative_prompt="background, scenery, landscape, ground, frame, border, pattern, color, gray, "
                                      "shading, hatching, texture, text, watermark",
                      num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
            la = normalise_lineart(np.asarray(im, np.float32) / 255)
            ink = np.asarray(la) < 128
            b = ink.shape[0] // 8
            border = np.concatenate([ink[:b].ravel(), ink[-b:].ravel(), ink[:, :b].ravel(), ink[:, -b:].ravel()]).mean()
            score = border + 0.5 * max(0.0, ink.mean() - 0.08)
            if best is None or score < best[0]:
                best = (score, seed, im, la)
        cid = "ood_" + s.split(" ", 1)[1].replace(" ", "_")
        best[2].save(OUT / f"{cid}_raw.jpg")
        best[3].save(OUT / f"{cid}.png")
        meta.append({"id": cid, "set": "ood", "subject": s, "seed": best[1], "border_ink": round(float(best[0]), 4)})
    (OUT / "contents.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(len(meta), "contents")


if __name__ == "__main__":
    main()
