"""All four SDXL line-art candidates per out-of-domain subject (review #11: unfiltered inputs).

Uses exactly the prompts and seeds of make_contents.py, so the selected candidate of each
subject is reproduced; 'selected' marks it.
"""
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

import make_contents as mc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/contents_oodall"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = {c["id"]: c["seed"] for c in json.loads((ROOT / "outputs/contents/contents.json").read_text()) if c["set"] == "ood"}
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    meta = []
    for i, s in enumerate(mc.OOD):
        base = "ood_" + s.split(" ", 1)[1].replace(" ", "_")
        for j in range(4):
            seed = 100 + 10 * i + j
            g = torch.Generator("cuda").manual_seed(seed)
            im = pipe(f"simple minimal black line drawing of {s}, single isolated subject, centered, "
                      f"coloring book outline, bold clean black outlines, lots of empty white space around it",
                      negative_prompt="background, scenery, landscape, ground, frame, border, pattern, color, gray, "
                                      "shading, hatching, texture, text, watermark",
                      num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
            cid = f"{base}_c{j}"
            mc.normalise_lineart(np.asarray(im, np.float32) / 255).save(OUT / f"{cid}.png")
            meta.append({"id": cid, "set": "ood_all", "subject": s, "seed": seed, "selected": seed == sel[base]})
    (OUT / "contents.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), sum(m["selected"] for m in meta))


if __name__ == "__main__":
    main()
