"""Seeded, fully logged B-LoRA generation (wraps the unmodified vendor/B-LoRA code).

Unlike vendor/B-LoRA/inference.py this fixes seeds, steps, CFG and resolution,
and writes everything needed to reproduce and evaluate a run:

    runs/<name>/images/<job>_s<seed>.png
    runs/<name>/run.json      arguments, adapter SHA-256, package versions, GPU, git commit
    runs/<name>/pairs.csv     input for tools/evaluate.py

Jobs CSV columns: `id,prompt` plus optional `content_image,style_image`
(reference images used only for evaluation; paths relative to the CSV).

    python tools/generate.py --jobs jobs.csv --name demo \
        --content_B_LoRA Experiment/test-round2/ksl_content.safetensors \
        --style_B_LoRA Experiment/test-round1/ksl_style.safetensors \
        --seeds 0 1 2 3 --steps 50 --cfg 5.0 --size 1024

Requires a CUDA GPU and the environment in vendor/B-LoRA/requirements.txt.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "B-LoRA"))


def sha256(path):
    if Path(path).stat().st_size < 1024 and Path(path).read_bytes().startswith(b"version https://git-lfs"):
        raise SystemExit(f"{path} is a Git LFS pointer; run: git lfs pull --include=\"{path}\" --exclude=\"\"")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_jobs(path):
    base = path.parent
    with path.open(encoding="utf-8-sig", newline="") as f:
        jobs = list(csv.DictReader(f))
    for j in jobs:
        for k in ("content_image", "style_image"):
            if j.get(k):
                j[k] = (base / j[k]).resolve()
    return jobs


def build_pipeline(args):
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from blora_utils import BLOCKS, filter_lora, scale_lora

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(args.base, vae=vae, torch_dtype=torch.float16).to("cuda")
    merged = {}
    for path, blocks, alpha in ((args.content_B_LoRA, BLOCKS["content"], args.content_alpha),
                                (args.style_B_LoRA, BLOCKS["style"], args.style_alpha)):
        if path:
            sd, _ = pipe.lora_state_dict(str(path))
            merged.update(scale_lora(filter_lora(sd, blocks), alpha))
    if merged:
        pipe.load_lora_into_unet(merged, None, pipe.unet)
    return pipe


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        import diffusers
        info.update(torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out_root", type=Path, default=ROOT / "runs")
    p.add_argument("--content_B_LoRA", type=Path)
    p.add_argument("--style_B_LoRA", type=Path)
    p.add_argument("--content_alpha", type=float, default=1.0)
    p.add_argument("--style_alpha", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--dry_run", action="store_true", help="write run.json/pairs.csv plan without generating")
    args = p.parse_args()

    out = args.out_root / args.name
    if out.exists() and any(out.iterdir()) and not args.dry_run:
        raise SystemExit(f"{out} exists; choose a new --name so earlier runs are never overwritten")
    (out / "images").mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.jobs)

    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "adapters": {k: {"path": str(v), "sha256": sha256(v)} for k, v in
                     (("content", args.content_B_LoRA), ("style", args.style_B_LoRA)) if v},
        "blocks": {"content": "unet.up_blocks.0.attentions.0", "style": "unet.up_blocks.0.attentions.1"},
        "git_commit": git_commit(),
        "environment": environment(),
        "jobs": [{k: str(v) for k, v in j.items()} for j in jobs],
    }
    rows = []
    pipe = None if args.dry_run else build_pipeline(args)
    for job in jobs:
        for seed in args.seeds:
            name = f"images/{job['id']}_s{seed}.png"
            if pipe is not None:
                import torch
                g = torch.Generator("cuda").manual_seed(seed)
                image = pipe(job["prompt"], negative_prompt=args.negative_prompt or None,
                             num_inference_steps=args.steps, guidance_scale=args.cfg,
                             height=args.size, width=args.size, generator=g).images[0]
                image.save(out / name)
            if job.get("content_image") and job.get("style_image"):
                rows.append({"id": f"{job['id']}_s{seed}", "content": job["content_image"],
                             "style": job["style_image"], "output": (out / name).resolve(), "prompt": job["prompt"]})
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if rows:
        with (out / "pairs.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["id", "content", "style", "output", "prompt"])
            w.writeheader()
            w.writerows(rows)
    print(f"{'planned' if args.dry_run else 'generated'} {len(jobs) * len(args.seeds)} images in {out}")


if __name__ == "__main__":
    main()
