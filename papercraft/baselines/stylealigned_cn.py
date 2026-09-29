"""StyleAligned (Hertz et al., CVPR 2024) baseline with the drawing supplied through the same canny ControlNet.

Official implementation (google/style-aligned @2faf0af, Apache 2.0) cloned to ~/third_party/style-aligned:
the Ku Shulan reference is DDIM-inverted (50 steps, CFG 2), shared attention with AdaIN on queries/keys
is registered as in the official real-image transfer notebook (shift log 2, scale 1), and the reference
branch follows its inversion trajectory (offset 5). The reference branch receives no ControlNet residual
(as in the official controlnet_call); the target branch receives the line drawing at scale 0.6, like all
ControlNet methods in this study. Same two style references and text information as InstantStyle.
Outputs: <cid>_s<k>_seed0.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, StableDiffusionXLControlNetPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papercraft"))
sys.path.insert(0, str(Path.home() / "third_party/style-aligned"))
import inversion  # noqa: E402
import priors  # noqa: E402
import sa_handler  # noqa: E402
from sample import control_image  # noqa: E402

REF_SUBJECT = {0: "an animal", 1: "a folk figure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/gen/stylealigned"))
    ap.add_argument("--styles", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--cn_scale", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False, set_alpha_to_one=False)
    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae, scheduler=sched,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # no ControlNet residual for the reference branch (batch under CFG: [u_ref, u_tgt, c_ref, c_tgt])
    orig_forward = pipe.controlnet.forward

    def cn_forward(*args, **kw):
        down, mid = orig_forward(*args, **kw)[:2]
        n = mid.shape[0]
        mask = torch.ones(n, device=mid.device, dtype=mid.dtype)
        mask[0::2] = 0
        m = lambda x: x * mask.view(-1, *([1] * (x.ndim - 1)))
        return [m(d) for d in down], m(mid)
    pipe.controlnet.forward = cn_forward

    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    contents = json.loads((ROOT / "outputs/contents/contents.json").read_text())
    if a.limit:
        contents = contents[: a.limit]
    for k in a.styles:
        ref = (priors.load_rgb(ROOT / split["style_refs"][k], 1024) * 255).astype(np.uint8)
        ref_prompt = f"{REF_SUBJECT[k]}, papercut"
        zts = inversion.ddim_inversion(pipe, ref, ref_prompt, a.steps, 2)
        handler = sa_handler.Handler(pipe)
        handler.register(sa_handler.StyleAlignedArgs(
            share_group_norm=True, share_layer_norm=True, share_attention=True, adain_queries=True,
            adain_keys=True, adain_values=False, shared_score_shift=np.log(2), shared_score_scale=1.0))
        for c in contents:
            f = out / f"{c['id']}_s{k}_seed0.png"
            if f.exists():
                continue
            ctrl, _ = control_image(ROOT / f"outputs/contents/{c['id']}.png")
            zT, cb = inversion.make_inversion_callback(zts, offset=5)
            g = torch.Generator("cpu").manual_seed(0)
            lat = torch.randn(2, 4, 128, 128, generator=g, dtype=torch.float16).to("cuda")
            lat[0] = zT
            imgs = pipe([ref_prompt, f"{c['subject']}, papercut"], image=[ctrl, ctrl], latents=lat,
                        controlnet_conditioning_scale=a.cn_scale, callback_on_step_end=cb,
                        num_inference_steps=a.steps, guidance_scale=a.cfg).images
            imgs[1].save(f)
            print("generated", f.name, flush=True)
        handler.remove()
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
