"""Secondary measures matching Zou et al. (npj Herit. Sci. 13:579, 2025), for the compact comparison.

content_clip  cosine similarity between CLIP ViT-L/14 image embeddings of the output and the line drawing
              (they report an L2-based "content preservation"; we use cosine and state it)
style_gram    VGG-19 Gram similarity to the style references: for relu1_1, relu2_1, relu3_1, relu4_1, relu5_1,
              cosine similarity between vectorised, size-normalised Gram matrices, averaged over layers and
              over the two references used by the reference-based methods
clip_s        cosine similarity between the image embedding and the text "a Ku Shulan style colour paper-cut"
lpips_content LPIPS (AlexNet) between output and drawing, both at 256 px (lower = closer to the drawing)
All at 512 px unless stated; appended as columns to <method>/rev_metrics.csv.
Usage: python ref_metrics.py <method> [<method> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEXT = "a Ku Shulan style colour paper-cut"
LAYERS = (1, 6, 11, 20, 29)  # relu1_1 ... relu5_1 in torchvision's vgg19.features


def load(path, size=512):
    return torch.from_numpy(priors.load_rgb(path, size)).permute(2, 0, 1)[None].float()


class Gram:
    def __init__(self):
        from torchvision.models import VGG19_Weights, vgg19
        self.net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        x = (x.to(DEV) - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in LAYERS:
                b, c, h, w = x.shape
                f = x.view(b, c, h * w)
                out.append(F.normalize((f @ f.transpose(1, 2) / (c * h * w)).flatten(1), dim=1))
        return out


def main(methods):
    import lpips
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(DEV)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV)
    gram = Gram()
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    ref_grams = [gram(load(ROOT / split["style_refs"][k])) for k in (0, 1)]

    @torch.no_grad()
    def img_emb(x):
        pil = [Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for t in x]
        return F.normalize(clip.get_image_features(**proc(images=pil, return_tensors="pt").to(DEV)), dim=-1)

    with torch.no_grad():
        t = F.normalize(clip.get_text_features(**proc(text=[TEXT], return_tensors="pt", padding=True).to(DEV)), dim=-1)
    for m in methods:
        f = G / m / "rev_metrics.csv"
        df = pd.read_csv(f)
        rows = []
        for name, cid in zip(df.file, df.content):
            x = load(G / m / name)
            d = load(ROOT / f"outputs/contents/{cid}.png")
            ex, ed = img_emb(x), img_emb(d)
            gx = gram(x)
            sg = np.mean([np.mean([float((a * b).sum()) for a, b in zip(gx, rg)]) for rg in ref_grams])
            with torch.no_grad():
                l = float(lp(F.interpolate(x, 256).to(DEV) * 2 - 1, F.interpolate(d, 256).to(DEV) * 2 - 1))
            rows.append({"content_clip": float((ex * ed).sum()), "style_gram": sg,
                         "clip_s": float((ex * t).sum()), "lpips_content": l})
        extra = pd.DataFrame(rows)
        for c in extra:
            df[c] = extra[c].values
        df.to_csv(f, index=False)
        print(m, extra.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
