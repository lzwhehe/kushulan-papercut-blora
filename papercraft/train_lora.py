"""LoRA training for SDXL attention blocks: B-LoRA baseline and craft-regularised style block.

Baseline (reproduces the official B-LoRA recipe on modern diffusers + PEFT):
    python train_lora.py --images IMG --prompt "A [v12]" --blocks content,style --out DIR

Collection style block with craft regularisation (this work):
    python train_lora.py --split outputs/data/split.json --blocks style \
        --craft self,foreign --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny, ControlNetModel, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import priors
from energies import CraftEnergy, palette_project

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {"content": "up_blocks.0.attentions.0", "style": "up_blocks.0.attentions.1"}
STYLE_TOKEN = "ksl papercut style"

# generic subjects used only by the foreign-content craft regulariser; kept disjoint
# from every evaluation subject (see make_contents.py)
FOREIGN_SUBJECTS = ["a hen", "a horse", "a sparrow", "an owl", "a goat", "a butterfly", "a peony",
                    "a lotus flower", "a pomegranate", "a teacup", "a lantern", "a vase",
                    "a girl with braids", "an old farmer", "a house", "a pine tree"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--prompt", default=None, help="single prompt for all --images")
    ap.add_argument("--split", default=None, help="split.json: train on split['train'] with captions")
    ap.add_argument("--blocks", default="content,style")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="random horizontal flip")
    ap.add_argument("--craft", default="", help="comma list of {self,foreign,proj}")
    ap.add_argument("--lam_proj", type=float, default=0.05)
    ap.add_argument("--proj_t", type=int, nargs=2, default=[200, 600])
    ap.add_argument("--lam_craft", type=float, default=0.05)
    ap.add_argument("--lam_keep", type=float, default=1.0)
    ap.add_argument("--p_foreign", type=float, default=0.5)
    ap.add_argument("--craft_tmax", type=int, default=700)
    ap.add_argument("--palette", default=str(ROOT / "outputs/data/palette.json"))
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--controlnet", action="store_true",
                    help="train with the frozen canny ControlNet in the loop on cut-line maps")
    ap.add_argument("--cn_range", type=float, nargs=2, default=[0.4, 0.8])
    ap.add_argument("--cutline_mode", default="both", choices=["both", "dense", "coarse", "canny"],
                    help="which structure maps the ControlNet reads during training")
    return ap.parse_args()


def load_square(path, res):
    rgb = priors.load_rgb(path, res)
    return torch.from_numpy(rgb).permute(2, 0, 1) * 2 - 1


@torch.no_grad()
def encode_prompts(prompts, dev):
    tok1 = CLIPTokenizer.from_pretrained(SDXL, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(SDXL, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(SDXL, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16).to(dev)
    te2 = CLIPTextModelWithProjection.from_pretrained(SDXL, subfolder="text_encoder_2", variant="fp16",
                                                      torch_dtype=torch.float16).to(dev)
    out = {}
    for p in set(prompts):
        ids1 = tok1(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        ids2 = tok2(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        h1 = te1(ids1, output_hidden_states=True).hidden_states[-2]
        o2 = te2(ids2, output_hidden_states=True)
        out[p] = (torch.cat([h1, o2.hidden_states[-2]], -1), o2.text_embeds)
    del te1, te2
    torch.cuda.empty_cache()
    return out


def lora_targets(unet, blocks):
    """Attention projections of the chosen B-LoRA blocks, or of every attention layer ('all')."""
    names = []
    for n, _ in unet.named_modules():
        in_block = "all" in blocks or any(n.startswith(BLOCKS[b] + ".") for b in blocks)
        if in_block and (
                n.endswith(("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"))):
            names.append(n)
    return names


@torch.no_grad()
def make_foreign_latents(dev, n_per=1, seed=1234):
    """Sample generic SDXL images (no LoRA) whose content the style block must papercut-ify."""
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, vae=vae, torch_dtype=torch.float16, variant="fp16").to(dev)
    pipe.set_progress_bar_config(disable=True)
    lats, subj = [], []
    g = torch.Generator(dev).manual_seed(seed)
    for s in FOREIGN_SUBJECTS:
        for _ in range(n_per):
            z = pipe(f"{s}, simple composition, plain white background", num_inference_steps=25,
                     guidance_scale=6.0, generator=g, output_type="latent").images
            lats.append(z.cpu())
            subj.append(s)
    del pipe
    torch.cuda.empty_cache()
    return torch.cat(lats), subj


def main():
    a = parse()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False))
    blocks = a.blocks.split(",")
    craft = [c for c in a.craft.split(",") if c]

    # ---- data
    if a.split:
        items = json.loads(Path(a.split).read_text())["train"]
        data = [(str(ROOT / it["path"]), f"{it['caption']} in {STYLE_TOKEN}") for it in items]
    else:
        data = [(p, a.prompt) for p in a.images]
    foreign_lat, foreign_prompts = None, []
    if "foreign" in craft or "proj" in craft:
        foreign_lat, subj = make_foreign_latents(dev)
        foreign_prompts = [f"{s} in {STYLE_TOKEN}" for s in subj]
    emb = encode_prompts([p for _, p in data] + foreign_prompts, dev)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(dev)
    lat_cache = []
    with torch.no_grad():
        for p, _ in data:
            x = load_square(p, a.res)[None].to(dev, torch.float16)
            xs = [x] + ([x.flip(-1)] if a.flip else [])
            lat_cache.append([vae.encode(v).latent_dist.sample() * vae.config.scaling_factor for v in xs])
    ctrl_cache = []
    if a.controlnet:
        cdir = ROOT / "outputs/cutlines"
        for p, _ in data:
            stem = str(Path(p).relative_to(ROOT)).replace("/", "__")
            kinds = {"both": ("dense", "coarse"), "dense": ("dense",), "coarse": ("coarse",),
                     "canny": ("canny",)}[a.cutline_mode]
            ctrl_cache.append([torch.from_numpy(np.asarray(Image.open(cdir / f"{stem}__{k}.png").convert("L")) < 128)
                               for k in kinds])
    if "proj" not in craft:
        del vae
    torch.cuda.empty_cache()

    # ---- model
    unet = UNet2DConditionModel.from_pretrained(SDXL, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(dev)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    targets = lora_targets(unet, blocks)
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian", target_modules=targets))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    print(f"LoRA modules={len(targets)} params={sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sched = DDPMScheduler.from_pretrained(SDXL, subfolder="scheduler")
    acp = sched.alphas_cumprod.to(dev)
    scaler = torch.amp.GradScaler()
    tid = torch.tensor([[a.res, a.res, 0, 0, a.res, a.res]], device=dev, dtype=torch.float16)

    controlnet = None
    if a.controlnet:
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", variant="fp16",
                                                     torch_dtype=torch.float16).to(dev)
        controlnet.requires_grad_(False)
    energy = None
    if "proj" in craft:
        pal = priors.Palette.load(a.palette)
        pal_lab_t = torch.tensor(pal.lab_with_white, device=dev, dtype=torch.float32)
        pal_rgb_t = torch.tensor(np.clip(priors.color.lab2rgb(pal.lab_with_white[None])[0], 0, 1), device=dev,
                                 dtype=torch.float32)
    if "self" in craft or "foreign" in craft:
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to(dev)
        taesd.requires_grad_(False)
        pal = priors.Palette.load(a.palette)
        energy = CraftEnergy(pal.lab, taesd, size=None).to(dev)

    def unet_eps(zt, t, prompt, ctrl=None, cn_scale=1.0):
        h, pooled = emb[prompt]
        cond = {"text_embeds": pooled, "time_ids": tid}
        extra = {}
        if ctrl is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                down, mid = controlnet(zt, t, encoder_hidden_states=h, controlnet_cond=ctrl, conditioning_scale=cn_scale,
                                       added_cond_kwargs=cond, return_dict=False)
            extra = {"down_block_additional_residuals": down, "mid_block_additional_residual": mid}
        with torch.autocast("cuda", dtype=torch.float16):
            return unet(zt, t, encoder_hidden_states=h, added_cond_kwargs=cond, **extra).sample.float()

    def x0_from(zt, eps, t):
        ab = acp[t].view(-1, 1, 1, 1)
        return (zt.float() - (1 - ab).sqrt() * eps) / ab.sqrt()

    def craft_loss(x0, t):
        # random 64x64 latent crop (= 512 px) keeps TAESD decoding cheap
        i, j = random.randint(0, x0.shape[-2] - 64), random.randint(0, x0.shape[-1] - 64)
        e, terms, _ = energy(x0[..., i:i + 64, j:j + 64], return_terms=True)
        return e.mean(), terms

    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        k = random.randrange(len(data))
        vi = random.randrange(len(lat_cache[k]))  # 0 = original, 1 = flipped
        z0 = lat_cache[k][vi].float()
        noise = torch.randn_like(z0)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
        zt = sched.add_noise(z0, noise, t)
        ctrl, cn_scale = None, 1.0
        if a.controlnet:
            m = random.choice(ctrl_cache[k])
            m = m.flip(-1) if vi == 1 else m
            ctrl = m.to(dev, torch.float16)[None, None].expand(1, 3, -1, -1)  # white lines on black
            cn_scale = random.uniform(*a.cn_range)
        eps = unet_eps(zt.half(), t, data[k][1], ctrl, cn_scale)
        loss_mse = F.mse_loss(eps, noise)
        loss = loss_mse
        rec = {"step": step, "t": int(t), "mse": float(loss_mse)}
        if "self" in craft and int(t) < a.craft_tmax:
            lc, terms = craft_loss(x0_from(zt, eps, t), t)
            w = float(acp[t])
            loss = loss + a.lam_craft * w * lc
            rec.update(self_craft=float(lc), self_pal=float(terms["pal"].mean()), self_flat=float(terms["flat"].mean()))
        if "foreign" in craft and random.random() < a.p_foreign:
            j = random.randrange(len(foreign_lat))
            zf = foreign_lat[j:j + 1].to(dev).float()
            tf = torch.randint(300, a.craft_tmax, (1,), device=dev)
            ztf = sched.add_noise(zf, torch.randn_like(zf), tf)
            x0f = x0_from(ztf, unet_eps(ztf.half(), tf, foreign_prompts[j]), tf)
            lc, terms = craft_loss(x0f, tf)
            # keep the coarse layout of the foreign content (anti-collapse)
            keep = F.mse_loss(F.avg_pool2d(x0f, 16), F.avg_pool2d(zf, 16))
            loss = loss + a.lam_craft * lc + a.lam_craft * a.lam_keep * keep
            rec.update(for_craft=float(lc), for_pal=float(terms["pal"].mean()),
                       for_flat=float(terms["flat"].mean()), keep=float(keep))
        if "proj" in craft and random.random() < a.p_foreign:
            # craft-projection self-distillation: the target is the cut projection of the
            # model's own clean-image prediction on foreign content (no degenerate optimum)
            j = random.randrange(len(foreign_lat))
            zf = foreign_lat[j:j + 1].to(dev).float()
            tf = torch.randint(a.proj_t[0], a.proj_t[1], (1,), device=dev)
            ztf = sched.add_noise(zf, torch.randn_like(zf), tf)
            epsf = unet_eps(ztf.half(), tf, foreign_prompts[j])
            with torch.no_grad():
                x0f = x0_from(ztf, epsf.detach(), tf)
                img = (vae.decode((x0f / vae.config.scaling_factor).half()).sample.float() + 1) / 2
                small = F.interpolate(img.clamp(0, 1), size=(512, 512), mode="area")
                proj = palette_project(small, pal_lab_t, pal_rgb_t)
                proj = F.interpolate(proj, size=img.shape[-2:], mode="nearest")
                zstar = vae.encode((proj * 2 - 1).half()).latent_dist.mean.float() * vae.config.scaling_factor
                shift = float((zstar - x0f).pow(2).mean())
            # x0-space distillation towards the projected prediction
            lp = F.mse_loss(x0_from(ztf, epsf, tf), zstar)
            loss = loss + a.lam_proj * lp
            rec.update(proj=float(lp), proj_shift=shift)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        log.append(rec)
        if step % a.log_every == 0:
            recent = log[-a.log_every:]
            keys = dict.fromkeys(kk for r in recent for kk in r if kk not in ("step", "t"))
            summ = {kk: np.mean([r[kk] for r in recent if kk in r]) for kk in keys}
            print(f"step {step} {time.time() - t0:.0f}s " + " ".join(f"{kk}={v:.4f}" for kk, v in summ.items()), flush=True)
        if step % 500 == 0 or step == a.steps:
            sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            StableDiffusionXLPipeline.save_lora_weights(out / (f"step{step}" if step != a.steps else ""), unet_lora_layers=sd,
                                                        safe_serialization=True)
    with open(out / "train_log.jsonl", "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    (out / "done.json").write_text(json.dumps({"seconds": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}))
    print("done", time.time() - t0, "s peak", torch.cuda.max_memory_allocated() / 2**30, "GB")


if __name__ == "__main__":
    main()
