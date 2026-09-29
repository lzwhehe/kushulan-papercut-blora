"""Leave-own-group-out CLIP style (fix for a reference leak).

The 13 de-duplicated held-out references are catalogue entries of test works. For an output of an in-domain
drawing, the style score is therefore recomputed against the mean embedding of the references that are NOT in
the near-duplicate group of that drawing's own entry; new-subject and repository drawings use all 13.
Rewrites clip_style_dedup (and proj_clip_style_dedup where plans exist) in every <method>/rev_metrics.csv;
the previous values are kept as *_allrefs. Also writes outputs/audit/loo_refs.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"


def main(methods):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    refs = json.loads((ROOT / "outputs/audit/heldout_refs.json").read_text())
    groups = json.loads((ROOT / "outputs/audit/groups.json").read_text())
    entry_group = {p: g for g, members in groups.items() for p in members}
    audit = json.loads((ROOT / "outputs/audit/split_audit.json").read_text())["rows"]
    own = {}  # content id -> set of reference indices to exclude
    for a in audit:
        cid = "ind_" + a["test"]
        g = entry_group.get(a["matched_entry"])
        own[cid] = {i for i, r in enumerate(refs) if r == a["matched_entry"] or (g is not None and entry_group.get(r) == g)}
    (ROOT / "outputs/audit/loo_refs.json").write_text(json.dumps({k: sorted(v) for k, v in own.items()}, ensure_ascii=False, indent=1))

    def ims(paths):
        return [Image.fromarray((priors.load_rgb(p, 512) * 255).astype(np.uint8)) for p in paths]

    @torch.no_grad()
    def emb(images):
        return torch.cat([torch.nn.functional.normalize(clip.get_image_features(**proc(images=images[i:i + 32], return_tensors="pt")), dim=-1)
                          for i in range(0, len(images), 32)])

    R = emb(ims([ROOT / p for p in refs]))  # 13 x d
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json").lab_with_white

    def loo_scores(E, contents):
        out = []
        for e, cid in zip(E, contents):
            keep = [i for i in range(len(refs)) if i not in own.get(cid, set())]
            c = torch.nn.functional.normalize(R[keep].mean(0), dim=0)
            out.append(float(e @ c))
        return out

    for m in methods:
        f = G / m / "rev_metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "clip_style_allrefs" not in df:
            df["clip_style_allrefs"] = df["clip_style_dedup"]
        E = emb(ims([G / m / x for x in df.file]))
        df["clip_style_dedup"] = loo_scores(E, df.content)
        labs = [G / m / Path(x).with_suffix(".labels.npy") for x in df.file]
        if all(l.exists() for l in labs):
            if "proj_clip_style_allrefs" not in df and "proj_clip_style_dedup" in df:
                df["proj_clip_style_allrefs"] = df["proj_clip_style_dedup"]
            P = emb([Image.fromarray((np.clip(priors.color.lab2rgb(pal[np.load(l)][None])[0], 0, 1) * 255).astype(np.uint8)) for l in labs])
            df["proj_clip_style_dedup"] = loo_scores(P, df.content)
        df.to_csv(f, index=False)
        ind = df.content.str.startswith("ind_")
        print(f"{m:20s} style all-refs {df.clip_style_allrefs.mean():.3f} -> LOO {df.clip_style_dedup.mean():.3f} "
              f"(in-domain {df.clip_style_allrefs[ind].mean():.3f} -> {df.clip_style_dedup[ind].mean():.3f})", flush=True)


if __name__ == "__main__":
    ms = sys.argv[1:] or sorted(d.name for d in G.iterdir() if (d / "rev_metrics.csv").exists())
    main(ms)
