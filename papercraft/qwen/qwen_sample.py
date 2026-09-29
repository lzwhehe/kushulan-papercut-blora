"""Sampling with the Qwen-Image-Edit-2511 backbone (groups Q0-Q2 of the backbone experiment).

Q0: base model, instruction only.  Q1: + edit LoRA trained on cut-line pairs (--lora).
Q2: + CraftGuide (--guide). The guidance is the same wrapper as on SDXL: for the flow-matching Euler
scheduler, sample = (1 - s) x0 + s * noise and the model predicts v = noise - x0, so x0_hat = sample - s * v,
exactly the form used by CraftGuide (sample.py). The energy is evaluated on the Qwen VAE decode of x0_hat
(the full decoder, used as its own proxy) at 512 px. Outputs follow the SDXL naming
(<cid>_s_seed<k>.png) so that evaluate.py, revision_metrics.py and stats.py apply unchanged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import priors  # noqa: E402
from energies import CraftEnergy  # noqa: E402
from sample import CraftGuide, ground_mask  # noqa: E402
from prepare_edit_data import INSTRUCTION  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen-Image-Edit-2511"
REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"


class QwenLatentDecoder:
    """Adapter with the TAESD interface expected by CraftEnergy: decode(packed latents).sample in [-1, 1]."""

    def __init__(self, pipe, height=1024, width=1024, scale=1.0):
        # scale < 1 average-pools the latent before decoding (pilot runs on small GPUs only)
        self.pipe, self.h, self.w, self.scale = pipe, height, width, scale
        self.dtype = pipe.vae.dtype
        cfg = pipe.vae.config
        self.mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1)
        self.std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1)

    def decode(self, packed):
        p = self.pipe
        lat = p._unpack_latents(packed, self.h, self.w, p.vae_scale_factor).to(self.dtype)
        lat = lat * self.std.to(lat) + self.mean.to(lat)
        if self.scale != 1.0:
            k = int(round(1 / self.scale))
            lat = F.avg_pool2d(lat[:, :, 0], k)[:, :, None]
        img = p.vae.decode(lat, return_dict=False)[0][:, :, 0]

        class _Out:
            sample = img
        return _Out()


class FlowCraftGuide(CraftGuide):
    """CraftGuide with the gradient normalised over all non-batch dims (packed latents are [B, L, C])."""

    def step(self, model_output, timestep, sample, *args, **kw):
        sch = self.pipe.scheduler
        i = sch.step_index if sch.step_index is not None else 0
        n = len(sch.timesteps)
        if self.window[0] * n <= i < self.window[1] * n:
            sigma = sch.sigmas[i].to(sample.device, torch.float32)
            x0 = (sample.float() - sigma * model_output.float()).detach()
            x = x0.clone()
            for _ in range(self.n_iter):
                x.requires_grad_(True)
                with torch.enable_grad():
                    e, terms, _ = self.energy(x, self.line, self.ground, return_terms=True)
                    g, = torch.autograd.grad(e.sum(), x)
                dims = tuple(range(1, g.ndim))
                g = g / (g.pow(2).mean(dims, keepdim=True).sqrt() + 1e-8)
                x = (x - self.strength * g).detach()
            self.trace.append({"step": i, **{k: float(v.mean()) for k, v in terms.items()}})
            model_output = ((sample.float() - x) / sigma).to(model_output.dtype)
        return self._orig(model_output, timestep, sample, *args, **kw)


def load_pipe(lora=None, dev="cuda", quant4=False, offload=False):
    """quant4: NF4 transformer and text encoder + model CPU offload, for pilot runs on a 20 GB GPU only
    (the reported experiment runs in bf16 on a 96 GB GPU)."""
    from diffusers import QwenImageEditPlusPipeline
    if not quant4:
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16)
        if lora:
            pipe.load_lora_weights(lora, adapter_name="cutline")
        pipe.set_progress_bar_config(disable=True)
        if offload:  # same bf16 weights, streamed layer by layer to a small GPU (slow, numerically identical path)
            pipe.enable_sequential_cpu_offload()
            return pipe
        return pipe.to(dev)
    from diffusers import BitsAndBytesConfig as DBnb, QwenImageTransformer2DModel
    from transformers import BitsAndBytesConfig as TBnb, Qwen2_5_VLForConditionalGeneration
    tr = QwenImageTransformer2DModel.from_pretrained(
        MODEL, revision=REVISION, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        quantization_config=TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, revision=REVISION, transformer=tr, text_encoder=te,
                                                     torch_dtype=torch.bfloat16)
    if lora:
        pipe.load_lora_weights(lora, adapter_name="cutline")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--guide_strength", type=float, default=1.0)
    ap.add_argument("--guide_iter", type=int, default=2)
    ap.add_argument("--contents_json", default=str(ROOT / "outputs/contents/contents.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="first N drawings only (smoke tests)")
    ap.add_argument("--quant4", action="store_true", help="NF4 + CPU offload (20 GB pilot only)")
    ap.add_argument("--offload", action="store_true", help="bf16 with sequential CPU offload (small GPUs)")
    ap.add_argument("--guide_decode_scale", type=float, default=1.0, help="<1: pooled-latent decode (pilot only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(a), "model": MODEL, "revision": REVISION}, indent=1))
    pipe = load_pipe(a.lora, quant4=a.quant4, offload=a.offload)
    guide = None
    if a.guide:
        pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
        energy = CraftEnergy(pal.lab, QwenLatentDecoder(pipe, scale=a.guide_decode_scale), w_pal=1, w_flat=1, w_edge=1, w_ground=1, size=512).cuda()
        if a.quant4:
            pipe.vae.to("cuda")  # the guidance decodes with gradients outside the offload hooks
        guide = FlowCraftGuide(pipe, energy, a.guide_strength, a.guide_iter)
    contents = json.loads(Path(a.contents_json).read_text())
    cdir = Path(a.contents_json).parent
    if a.limit:
        contents = contents[: a.limit]
    records = []
    for c in contents:
        drawing = Image.open(cdir / f"{c['id']}.png").convert("RGB").resize((1024, 1024))
        line = torch.from_numpy(np.asarray(drawing.convert("L")) < 128)
        if guide:
            guide.line = line[None, None].cuda()
            guide.ground = ground_mask(line, 12)[None, None].cuda()
        for seed in a.seeds:
            f = out / f"{c['id']}_s_seed{seed}.png"
            if f.exists():
                continue
            if guide:
                guide.trace = []
            torch.cuda.synchronize(); t0 = time.perf_counter()
            img = pipe(image=[drawing], prompt=INSTRUCTION.format(subject=c["subject"]), negative_prompt=" ",
                       true_cfg_scale=a.cfg, num_inference_steps=a.steps, height=1024, width=1024,
                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            torch.cuda.synchronize()
            img.save(f)
            records.append({"file": f.name, "seconds": time.perf_counter() - t0,
                            "trace": guide.trace if guide else None})
            print("generated", f.name, "%.1fs" % records[-1]["seconds"], flush=True)
    with open(out / "records.jsonl", "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
