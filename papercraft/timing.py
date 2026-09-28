"""Review #35: per-drawing time breakdown of the full CutCraft pipeline on 10 drawings.

Stages: sampling (50 steps incl. CraftGuide), VAE decode is part of sampling; cut projection
(CPU, 512 px); SVG export (CPU, 1024 px labels). One warm-up run is excluded.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderTiny
from PIL import Image

import priors
import sample as S
from energies import CraftEnergy

ROOT = Path(__file__).resolve().parents[1]


def main():
    pipe = S.load_pipe(True)
    pipe.load_lora_weights(S.filtered_lora(ROOT / "outputs/style/cutline/pytorch_lora_weights.safetensors", 1),
                           adapter_name="style")
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).cuda()
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    energy = CraftEnergy(pal.lab, taesd, w_pal=1, w_flat=1, w_edge=1, w_ground=1, size=512).cuda()
    guide = S.CraftGuide(pipe, energy, 1.0, 2)
    contents = json.loads((ROOT / "outputs/contents/contents.json").read_text())[:11]
    rows = []
    for n, c in enumerate(contents):
        ctrl, line = S.control_image(ROOT / f"outputs/contents/{c['id']}.png")
        guide.line = line[None, None].cuda()
        guide.ground = S.ground_mask(line, 12)[None, None].cuda()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        img = pipe(prompt=f"{c['subject']} in ksl papercut style", image=ctrl, controlnet_conditioning_scale=0.6,
                   num_inference_steps=50, guidance_scale=5.0, generator=torch.Generator("cuda").manual_seed(0),
                   height=1024, width=1024).images[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        rgb = np.asarray(img.resize((512, 512), Image.LANCZOS), np.float32) / 255
        la = priors.load_rgb(ROOT / f"outputs/contents/{c['id']}.png", 512)
        g = priors.ground_from_lineart(la.mean(-1) < 0.85)
        _, labels, _, _ = priors.cut_project(rgb, pal, ground=g)
        t2 = time.perf_counter()
        priors.export_layers_svg(labels, pal, ROOT / "outputs/scratch/timing.svg")
        t3 = time.perf_counter()
        if n > 0:
            rows.append({"id": c["id"], "sampling_s": t1 - t0, "projection_s": t2 - t1, "svg_s": t3 - t2,
                         "total_s": t3 - t0})
    out = {k: {"median": float(np.median([r[k] for r in rows])), "min": float(np.min([r[k] for r in rows])),
               "max": float(np.max([r[k] for r in rows]))} for k in rows[0] if k != "id"}
    out["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    out["gpu"] = torch.cuda.get_device_name(0)
    (ROOT / "outputs/timing.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    (ROOT / "outputs/scratch").mkdir(parents=True, exist_ok=True)
    main()
