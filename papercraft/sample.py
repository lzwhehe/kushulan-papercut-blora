"""Generate papercuts for the evaluation contents with every compared method.

Methods
  blora            official B-LoRA recipe: content block from a line-art B-LoRA, style block
                   from a single-reference B-LoRA (per-content training required)
  prompt_cn        SDXL + canny ControlNet on the line art, descriptive papercut prompt, no LoRA
  blora_style_cn   ControlNet + style block of a single-reference B-LoRA
  coll_cn          ControlNet + collection style block (no craft regulariser)
  coll_self_cn     ControlNet + collection style block, self craft regulariser
  cutcraft_cn      ControlNet + collection style block, self + foreign craft regulariser
  prompt_only      SDXL with the descriptive papercut prompt only: no line constraint, no LoRA
  coll_nocn        a collection/cut-line style block (--lora) without ControlNet: no line constraint
Any method accepts --guide to add CraftGuide (sampling-time craft energy guidance).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (AutoencoderKL, AutoencoderTiny, ControlNetModel, StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline)
from PIL import Image

import priors
from energies import CraftEnergy

ROOT = Path(__file__).resolve().parents[1]
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_TOKEN = "ksl papercut style"
DESC = "Ku Shulan Chinese folk papercut, layered cut colored paper collage, flat colors, white background"


def filtered_lora(path, block):
    sd, _ = StableDiffusionXLPipeline.lora_state_dict(str(path))
    return {k: v for k, v in sd.items() if f"up_blocks.0.attentions.{block}." in k}


class CraftGuide:
    """Wraps scheduler.step: corrects the implied x0 towards low craft energy.

    With the Euler (k-diffusion) parameterisation sample = x0 + sigma * eps, so
    x0_hat = sample - sigma * eps. We take n normalised gradient steps on
    E(x0_hat) and hand the scheduler eps' = (sample - x0_hat') / sigma.
    """

    def __init__(self, pipe, energy, strength=0.3, n_iter=2, window=(0.15, 0.75)):
        self.pipe, self.energy = pipe, energy
        self.strength, self.n_iter, self.window = strength, n_iter, window
        self.line = None
        self.ground = None
        self._orig = pipe.scheduler.step
        pipe.scheduler.step = self.step
        self.trace = []

    def step(self, model_output, timestep, sample, *args, **kw):
        sch = self.pipe.scheduler
        i = sch.step_index if sch.step_index is not None else 0
        n = len(sch.timesteps)
        if self.window[0] * n <= i < self.window[1] * n:
            sigma = sch.sigmas[i].to(sample.device, torch.float32)
            eps = model_output.float()
            x0 = (sample.float() - sigma * eps).detach()
            x = x0.clone()
            for _ in range(self.n_iter):
                x.requires_grad_(True)
                with torch.enable_grad():
                    e, terms, _ = self.energy(x, self.line, self.ground, return_terms=True)
                    g, = torch.autograd.grad(e.sum(), x)
                g = g / (g.pow(2).mean((1, 2, 3), keepdim=True).sqrt() + 1e-8)
                x = (x - self.strength * g).detach()
            self.trace.append({"step": i, **{k: float(v.mean()) for k, v in terms.items()}})
            model_output = ((sample.float() - x) / sigma).to(model_output.dtype)
        return self._orig(model_output, timestep, sample, *args, **kw)


def load_pipe(controlnet: bool, dev="cuda"):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    if controlnet:
        cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16,
                                             variant="fp16")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(SDXL, vae=vae, controlnet=cn, variant="fp16",
                                                                   torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, vae=vae, variant="fp16", torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(dev)


def control_image(line_png):
    lines = np.asarray(Image.open(line_png).convert("L")) < 128
    return Image.fromarray((lines * 255).astype(np.uint8)).convert("RGB"), torch.from_numpy(lines)


def ground_mask(line: torch.Tensor, margin: int = 12) -> torch.Tensor:
    """Pixels outside the (dilated) silhouette of the line drawing, at 512 px."""
    from scipy import ndimage as ndi
    from skimage import morphology
    small = np.asarray(Image.fromarray(line.numpy().astype(np.uint8) * 255).resize((512, 512), Image.BILINEAR)) > 40
    sil = priors.silhouette_from_lineart(small)
    sil = ndi.binary_dilation(sil, morphology.disk(margin))
    return torch.from_numpy(~sil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--lora", default=None, help="collection style LoRA dir (coll_* / cutcraft_*)")
    ap.add_argument("--styles", type=int, nargs="*", default=[0, 1], help="style-reference ids (B-LoRA methods)")
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--guide_strength", type=float, default=0.3)
    ap.add_argument("--guide_iter", type=int, default=2)
    ap.add_argument("--w_pal", type=float, default=1.0)
    ap.add_argument("--w_flat", type=float, default=1.0)
    ap.add_argument("--w_edge", type=float, default=1.0)
    ap.add_argument("--w_ground", type=float, default=1.0)
    ap.add_argument("--ground_margin", type=int, default=12, help="px at 512 around the silhouette left free")
    ap.add_argument("--cn_end", type=float, default=1.0, help="fraction of steps with ControlNet active")
    ap.add_argument("--cn_scale", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--contents", nargs="*", default=None)
    ap.add_argument("--contents_json", default=str(ROOT / "outputs/contents/contents.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    contents = json.loads(Path(a.contents_json).read_text())
    if a.contents:
        contents = [c for c in contents if c["id"] in a.contents]
    all_ids = [c["id"] for c in json.loads(Path(a.contents_json).read_text())]

    use_cn = a.method not in ("blora", "prompt_only", "coll_nocn")
    pipe = load_pipe(use_cn)
    guide = None
    if a.guide:
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")
        pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
        energy = CraftEnergy(pal.lab, taesd, w_pal=a.w_pal, w_flat=a.w_flat, w_edge=a.w_edge, w_ground=a.w_ground,
                             size=512).cuda()
        guide = CraftGuide(pipe, energy, a.guide_strength, a.guide_iter)

    style_ids = a.styles if a.method in ("blora", "blora_style_cn", "instantstyle_cn") else [None]
    if a.method.startswith(("coll", "cutcraft")):
        sd = filtered_lora(Path(a.lora) / "pytorch_lora_weights.safetensors", 1)
        pipe.load_lora_weights(sd, adapter_name="style")
    elif a.method == "fulllora_cn":  # conventional LoRA on every attention layer
        pipe.load_lora_weights(str(Path(a.lora) / "pytorch_lora_weights.safetensors"), adapter_name="style")
    elif a.method == "instantstyle_cn":  # IP-Adapter injected only into the style block (InstantStyle)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.safetensors",
                             image_encoder_folder="models/image_encoder")
        pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, 1.0, 0.0]}})
        split = json.loads((ROOT / "outputs/data/split.json").read_text())
        ref_imgs = {k: Image.fromarray((priors.load_rgb(ROOT / split["style_refs"][k], 1024) * 255).astype(np.uint8))
                    for k in a.styles}

    records = []
    for c in contents:
        ctrl, line = control_image(Path(a.contents_json).parent / f"{c['id']}.png")
        if guide is not None:
            guide.line = line[None, None].cuda() if a.w_edge > 0 else None
            guide.ground = ground_mask(line, a.ground_margin)[None, None].cuda() if a.w_ground > 0 else None
        for k in style_ids:
            if a.method == "blora":
                ci = all_ids.index(c["id"])
                sd = {**filtered_lora(ROOT / f"outputs/blora/content_{c['id']}/pytorch_lora_weights.safetensors", 0),
                      **filtered_lora(ROOT / f"outputs/blora/style{k}/pytorch_lora_weights.safetensors", 1)}
                pipe.load_lora_weights(sd, adapter_name="blora")
                prompt = f"A [c{ci}] in [s{k}] style"
            elif a.method == "blora_style_cn":
                pipe.load_lora_weights(filtered_lora(ROOT / f"outputs/blora/style{k}/pytorch_lora_weights.safetensors", 1),
                                       adapter_name="blora")
                prompt = f"{c['subject']} in [s{k}] style"
            elif a.method == "instantstyle_cn":
                prompt = f"{c['subject']}, papercut"
            elif a.method in ("prompt_cn", "prompt_only"):
                prompt = f"{c['subject']}, {DESC}"
            else:
                prompt = f"{c['subject']} in {STYLE_TOKEN}"
            for seed in a.seeds:
                name = f"{c['id']}_s{'' if k is None else k}_seed{seed}"
                if (out / f"{name}.png").exists():
                    continue
                g = torch.Generator("cuda").manual_seed(seed)
                kw = dict(prompt=prompt, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g,
                          height=1024, width=1024)
                if use_cn:
                    kw.update(image=ctrl, controlnet_conditioning_scale=a.cn_scale, control_guidance_end=a.cn_end)
                if a.method == "instantstyle_cn":
                    kw.update(ip_adapter_image=ref_imgs[k])
                if guide is not None:
                    guide.trace = []
                t0 = time.time()
                img = pipe(**kw).images[0]
                img.save(out / f"{name}.png")
                records.append({"name": name, "content": c["id"], "style": k, "seed": seed, "prompt": prompt,
                                "seconds": time.time() - t0, "guide_trace": guide.trace if guide else None})
            if a.method in ("blora", "blora_style_cn"):
                pipe.unload_lora_weights()
    with open(out / "records.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("generated", len(records))


if __name__ == "__main__":
    main()
