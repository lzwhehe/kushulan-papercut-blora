"""Score generated papercuts: craft, structure, style and leakage metrics.

python evaluate.py --gen outputs/gen/<method> [--project]
writes <gen>/metrics.csv (and metrics_projected.csv with --project, where every
output is first passed through the cut projection).
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import morphology
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

import priors

ROOT = Path(__file__).resolve().parents[1]
_PAL = None


def _pal():
    global _PAL
    if _PAL is None:
        _PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")
    return _PAL


def _init_worker():
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def cpu_metrics(args):
    path, content_id, project, cdir = args
    rgb = priors.load_rgb(path, 512)
    rec = {}
    la = priors.load_rgb(Path(cdir) / f"{content_id}.png", 512)
    lines = la.mean(-1) < 0.85
    if project:
        rgb, labels, _, st = priors.cut_project(rgb, _pal(), ground=priors.ground_from_lineart(lines))
        rec.update(layers=st["layers"])
        np.save(Path(path).with_suffix(".labels.npy"), labels.astype(np.uint8))
    rec.update(priors.craft_metrics(rgb, _pal()))
    rec.update(priors.edge_f1(priors.edge_map(rgb), lines, tol=3))
    rec["sil_iou"] = priors.silhouette_iou(rgb, priors.silhouette_from_lineart(lines))
    return rec


class Embedder:
    def __init__(self, dev="cuda"):
        self.dev = dev
        self.dt = torch.float16 if dev == "cuda" else torch.float32
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dt).to(dev).eval()
        self.cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
        self.dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    @torch.no_grad()
    def images(self, ims):
        c, d = [], []
        for i in range(0, len(ims), 16):
            b = ims[i:i + 16]
            x = self.cproc(images=b, return_tensors="pt")["pixel_values"].to(self.dev, self.dt)
            c.append(torch.nn.functional.normalize(self.clip.get_image_features(pixel_values=x).float(), dim=-1))
            x = self.dproc(images=b, return_tensors="pt").to(self.dev)
            d.append(torch.nn.functional.normalize(self.dino(**x).pooler_output.float(), dim=-1))
        return torch.cat(c).cpu(), torch.cat(d).cpu()

    @torch.no_grad()
    def texts(self, txt):
        x = self.cproc(text=txt, return_tensors="pt", padding=True).to(self.dev)
        return torch.nn.functional.normalize(self.clip.get_text_features(**x).float(), dim=-1).cpu()


def pil(path, size=512):
    return Image.fromarray((priors.load_rgb(path, size) * 255).astype(np.uint8))


def kid(x, y, n_sub=50, sub=64, seed=0):
    """Kernel Inception Distance with a cubic polynomial kernel on given features (x1000)."""
    rng = np.random.default_rng(seed)
    x, y = x.numpy().astype(np.float64), y.numpy().astype(np.float64)
    d = x.shape[1]
    m = min(sub, len(x), len(y))
    vals = []
    for _ in range(n_sub):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        kxx = (a @ a.T / d + 1) ** 3
        kyy = (b @ b.T / d + 1) ** 3
        kxy = (a @ b.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals) * 1000), float(np.std(vals) * 1000)


_REF = None


def references(emb):
    """Held-out Ku Shulan works (never used for training) and the full corpus."""
    global _REF
    if _REF is None:
        split = json.loads((ROOT / "outputs/data/split.json").read_text())
        held = split["held_out"] + [t["reference"] for t in split["test_indomain"]]
        corpus = [it["path"] for it in split["train"]] + split["held_out"]
        hc, _ = emb.images([pil(ROOT / p) for p in held])
        cc, _ = emb.images([pil(ROOT / p) for p in corpus])
        _, sd = emb.images([pil(ROOT / p) for p in split["style_refs"]])
        gt = {"ind_" + t["id"]: emb.images([pil(ROOT / t["reference"])]) for t in split["test_indomain"]}
        _REF = {"held_centroid": torch.nn.functional.normalize(hc.mean(0), dim=0), "corpus": cc,
                "style_dino": sd, "gt": gt}
    return _REF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--contents_json", default=str(ROOT / "outputs/contents/contents.json"))
    a = ap.parse_args()
    contents = {c["id"]: c for c in json.loads(Path(a.contents_json).read_text())}
    cdir = str(Path(a.contents_json).parent)
    import multiprocessing as mp
    jobs = {}
    for gdir in map(Path, a.gen):
        parsed = []
        for f in sorted(gdir.glob("*.png")):
            m = re.match(r"(.+)_s(\d*)_seed(\d+)$", f.stem)
            parsed.append((f, m.group(1), m.group(2), int(m.group(3))))
        jobs[gdir] = parsed
    flat = [(str(f), cid, a.project, cdir) for p in jobs.values() for f, cid, _, _ in p]
    with ProcessPoolExecutor(24, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        res = list(ex.map(cpu_metrics, flat, chunksize=2))
    cpu_all, i0 = {}, 0
    for gdir, p in jobs.items():
        cpu_all[gdir] = res[i0:i0 + len(p)]
        i0 += len(p)
    emb = Embedder(a.device)
    ref = references(emb)
    for gdir, parsed in jobs.items():
        cpu = cpu_all[gdir]
        ims = []
        for f, cid, _, _ in parsed:
            rgb = priors.load_rgb(f, 512)
            if a.project:
                lab = np.load(f.with_suffix(".labels.npy"))
                pl = _pal().lab_with_white
                rgb = np.clip(priors.color.lab2rgb(pl[lab][None])[0], 0, 1)
            ims.append(Image.fromarray((rgb * 255).astype(np.uint8)))
        ce, de = emb.images(ims)
        te = emb.texts([f"a papercut of {contents[cid]['subject']}" for _, cid, _, _ in parsed])
        rows = []
        for i, (f, cid, sk, seed) in enumerate(parsed):
            r = {"file": f.name, "content": cid, "set": contents[cid]["set"], "style": sk, "seed": seed, **cpu[i]}
            r["clip_style"] = float(ce[i] @ ref["held_centroid"])
            r["clip_text"] = float(ce[i] @ te[i])
            # leakage: DINO similarity to the single-image style references
            r["dino_styleref"] = float((de[i] @ ref["style_dino"][:2].T).max())
            if cid in ref["gt"]:
                gc, gd = ref["gt"][cid]
                r["dino_gt"] = float(de[i] @ gd[0])
                r["clip_gt"] = float(ce[i] @ gc[0])
            rows.append(r)
        df = pd.DataFrame(rows)
        name = "metrics_projected.csv" if a.project else "metrics.csv"
        df.to_csv(gdir / name, index=False)
        k, ks = kid(ce, ref["corpus"])
        (gdir / name.replace(".csv", "_kid.json")).write_text(json.dumps({"clip_kid_x1000": k, "std": ks, "n": len(df)}))
        print(gdir.name, len(df), "KID", round(k, 2))


if __name__ == "__main__":
    main()
