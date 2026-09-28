"""Review #28: do craft energies on the TAESD proxy decode agree with the final VAE decode?

For 40 CutCraft and 40 cut-line outputs: encode the final image with the SDXL VAE, decode the
latent with TAESD (as CraftGuide does) and with the VAE, and compare each energy term.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny
from scipy.stats import pearsonr, spearmanr

import priors
import sample as S
from energies import CraftEnergy

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).cuda()
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).cuda()
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    energy = CraftEnergy(pal.lab, taesd, size=512).cuda()
    rows = []
    for m in ("cutcraft", "cutline"):
        for f in sorted((ROOT / f"outputs/gen/{m}").glob("*_seed0.png"))[:40]:
            cid = f.name.split("_s_seed")[0]
            _, line = S.control_image(ROOT / f"outputs/contents/{cid}.png")
            line = line[None, None].cuda()
            ground = S.ground_mask(line[0, 0].cpu(), 12)[None, None].cuda()
            x = torch.from_numpy(priors.load_rgb(f, 1024)).permute(2, 0, 1)[None].cuda().half() * 2 - 1
            z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            proxy = energy.decode(z)
            full = (vae.decode(z / vae.config.scaling_factor).sample.float() + 1) / 2
            full = F.interpolate(full.clamp(0, 1), size=(512, 512), mode="bilinear", antialias=True)
            tp = energy.terms(proxy, line, ground)
            tf = energy.terms(full, line, ground)
            rows.append({"method": m, **{f"proxy_{k}": float(v) for k, v in tp.items()},
                         **{f"vae_{k}": float(v) for k, v in tf.items()}})
    out = {}
    for k in ("pal", "flat", "edge", "ground"):
        a = np.array([r[f"proxy_{k}"] for r in rows]); b = np.array([r[f"vae_{k}"] for r in rows])
        out[k] = {"pearson": pearsonr(a, b)[0], "spearman": spearmanr(a, b)[0],
                  "mean_proxy": a.mean(), "mean_vae": b.mean(), "median_abs_diff": float(np.median(np.abs(a - b)))}
    (ROOT / "outputs/decoder_check.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
