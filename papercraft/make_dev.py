"""Three dev line arts (disjoint from test) used only for hyper-parameter selection."""
import json
import make_contents as mc

mc.OOD = ["a lion", "a peach branch", "a lady holding a fan"]
OUT = mc.OUT.parent / "contents_dev"
mc.OUT = OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import numpy as np, torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    meta = []
    for i, s in enumerate(mc.OOD):
        g = torch.Generator("cuda").manual_seed(900 + i)
        im = pipe(f"simple minimal black line drawing of {s}, single isolated subject, centered, coloring book "
                  f"outline, bold clean black outlines, lots of empty white space around it",
                  negative_prompt="background, scenery, landscape, ground, frame, border, pattern, color, gray, "
                                  "shading, hatching, texture, text, watermark",
                  num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
        cid = "dev_" + s.split(" ", 1)[1].replace(" ", "_")
        mc.normalise_lineart(np.asarray(im, np.float32) / 255).save(OUT / f"{cid}.png")
        meta.append({"id": cid, "set": "dev", "subject": s})
    (OUT / "contents.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
