"""Record x0-hat and craft energies along the sampling trajectory (Fig. 5).

Runs the cut-line model on one drawing twice (with / without CraftGuide), and at every
step stores the four energy terms of the x0 that is handed to the scheduler; at a few
steps it also decodes x0 with TAESD for display.
"""
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderTiny
from PIL import Image

import priors
import sample as S
from energies import CraftEnergy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/assets"
SHOW = [5, 12, 20, 30, 45]


def run(guided, cid="ood_rooster", seed=0, save=True):
    pipe = S.load_pipe(True)
    pipe.load_lora_weights(S.filtered_lora(ROOT / "outputs/style/cutline/pytorch_lora_weights.safetensors", 1),
                           adapter_name="style")
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).cuda()
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    energy = CraftEnergy(pal.lab, taesd, w_pal=1, w_flat=1, w_edge=1, w_ground=1, size=512).cuda()
    ctrl, line = S.control_image(ROOT / f"outputs/contents/{cid}.png")
    guide = S.CraftGuide(pipe, energy, 1.0 if guided else 0.0, 2)
    guide.line = line[None, None].cuda()
    guide.ground = S.ground_mask(line, 12)[None, None].cuda()
    inner = pipe.scheduler.step  # CraftGuide.step
    log = []

    def rec_step(model_output, timestep, sample, *a, **k):
        sch = pipe.scheduler
        i = sch.step_index if sch.step_index is not None else 0
        out = inner(model_output, timestep, sample, *a, **k)
        # recompute the x0 that was used: sample - sigma * eps' (eps' from the guided output)
        sigma = sch.sigmas[i].to(sample.device, torch.float32)
        prev = (out[0] if isinstance(out, tuple) else out.prev_sample).float()
        # Euler: prev = sample + (sigma_next - sigma) * d, d = (sample - x0)/sigma
        sn = sch.sigmas[i + 1].to(sample.device, torch.float32)
        d = (prev - sample.float()) / (sn - sigma)
        x0 = sample.float() - sigma * d
        with torch.no_grad():
            e, t, rgb = energy(x0, guide.line, guide.ground, return_terms=True)
        log.append({"step": i + 1, **{kk: float(v.mean()) for kk, v in t.items()}, "E": float(e.mean())})
        if save and i + 1 in SHOW:
            Image.fromarray((rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
                OUT / f"traj_{'g' if guided else 'u'}_{i + 1:02d}.jpg", quality=92)
        return out

    pipe.scheduler.step = rec_step
    g = torch.Generator("cuda").manual_seed(seed)
    subj = {c["id"]: c["subject"] for c in json.loads((ROOT / "outputs/contents/contents.json").read_text())}[cid]
    img = pipe(prompt=f"{subj} in ksl papercut style", image=ctrl, controlnet_conditioning_scale=0.6,
               num_inference_steps=50, guidance_scale=5.0, generator=g, height=1024, width=1024).images[0]
    if save:
        img.resize((512, 512)).save(OUT / f"traj_{'g' if guided else 'u'}_final.jpg", quality=92)
    del pipe
    torch.cuda.empty_cache()
    return log


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":  # review #28: several drawings, no images saved
        ids = ["ood_rooster", "ood_tiger", "ood_teapot", "ind_人物-5", "ind_植物-29", "repo_fish"]
        res = {cid: {"unguided": run(False, cid, save=False), "guided": run(True, cid, save=False)} for cid in ids}
        (ROOT / "outputs/trajectory_multi.json").write_text(json.dumps(res, indent=1))
    else:
        res = {"unguided": run(False), "guided": run(True)}
        (ROOT / "outputs/trajectory.json").write_text(json.dumps(res, indent=1))
        print({k: v[-1] for k, v in res.items()})
