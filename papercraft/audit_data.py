"""Review #06-#08, #19: catalogue inventory, near-duplicate groups, split audit, reference sets.

Outputs (outputs/audit/):
  inventory.csv          one row per catalogue image entry: subset, category, split role, group id
  groups.json            near-duplicate groups (DINOv2 cos > 0.93 OR pHash Hamming <= 8, union-find)
  split_audit.json       per test work: its group, and its nearest training images below the threshold
  nn_review_*.jpg        contact sheets of each test work and its 3 nearest training images (visual review)
  ood_subject_nn.json    for each out-of-domain subject: CLIP text->image top-5 training images
  ood_subject_nn.jpg     contact sheet for visual review
  heldout_refs.json      de-duplicated held-out reference set (one image per group, never trained)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

import priors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit"
REP = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
PAT = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Pattern_symbols"


def phash(img, size=32, keep=8):
    from scipy.fft import dctn
    g = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), np.float32)
    d = dctn(g, norm="ortho")[:keep, :keep]
    return (d > np.median(d)).ravel()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    rep = sorted(str(Path(p).relative_to(ROOT)) for p in REP.glob("*/*"))
    pat = sorted(str(Path(p).relative_to(ROOT)) for p in PAT.glob("*/*"))
    cat = rep + pat
    train = {it["path"] for it in split["train"]}
    held = set(split["held_out"])
    tests = split["test_indomain"]
    imgs = [Image.fromarray((priors.load_rgb(ROOT / p, 256) * 255).astype(np.uint8)) for p in cat]
    test_imgs = [Image.fromarray((priors.load_rgb(ROOT / t["reference"], 256) * 255).astype(np.uint8)) for t in tests]

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    def emb(ims):
        out = []
        with torch.no_grad():
            for i in range(0, len(ims), 32):
                x = proc(images=ims[i:i + 32], return_tensors="pt")
                out.append(torch.nn.functional.normalize(dino(**x).pooler_output, dim=-1))
        return torch.cat(out)
    E, Et = emb(imgs), emb(test_imgs)
    S = (E @ E.T).numpy()
    H = np.stack([phash(im) for im in imgs])
    ham = (H[:, None, :] != H[None, :, :]).sum(-1)

    # union-find near-duplicate groups over the catalogue
    parent = list(range(len(cat)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            if S[i, j] > 0.93 or ham[i, j] <= 8:
                parent[find(i)] = find(j)
    gid = [find(i) for i in range(len(cat))]
    groups = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(cat[i])
    multi = {str(k): v for k, v in groups.items() if len(v) > 1}

    # split audit: test works vs training images
    St = (Et @ E.T).numpy()
    audit, bad = [], 0
    train_idx = [i for i, p in enumerate(cat) if p in train]
    for t, s in zip(tests, St):
        order = [i for i in np.argsort(-s) if cat[i] in train][:3]
        match = int(np.argmax(s))
        g = gid[match]
        shared = [p for p in groups[g] if p in train]
        bad += bool(shared)
        audit.append({"test": t["id"], "matched_entry": cat[match], "match_sim": round(float(s[match]), 3),
                      "group_size": len(groups[g]), "training_members_in_group": shared,
                      "nearest_training": [(cat[i], round(float(s[i]), 3)) for i in order]})
    # contact sheets of test work + 3 nearest training images
    W = 150
    for part in range(0, len(tests), 10):
        rows = audit[part:part + 10]
        sheet = Image.new("RGB", (W * 4, (W + 16) * len(rows)), "white")
        dr = ImageDraw.Draw(sheet)
        for r, a in enumerate(rows):
            ims = [test_imgs[part + r]] + [imgs[cat.index(p)] for p, _ in a["nearest_training"]]
            labs = [a["test"][:14]] + [f"{sim:.2f}" for _, sim in a["nearest_training"]]
            for c, (im, lab) in enumerate(zip(ims, labs)):
                sheet.paste(im.resize((W, W)), (c * W, r * (W + 16) + 16))
                dr.text((c * W + 3, r * (W + 16) + 2), lab, fill="black")
        sheet.save(OUT / f"nn_review_{part // 10}.jpg", quality=88)

    # de-duplicated held-out reference set: one representative per group, no group member trained
    held_groups = {}
    for p in held:
        g = gid[cat.index(p)]
        if not any(q in train for q in groups[g]):
            held_groups.setdefault(g, p)
    refs = sorted(held_groups.values())
    (OUT / "heldout_refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1))

    # inventory
    rows = []
    for i, p in enumerate(cat):
        role = "train" if p in train else "held_out" if p in held else "unused"
        rows.append({"entry": p, "subset": "representative" if p in rep else "pattern_symbol",
                     "category": Path(p).parent.name, "role": role, "group": gid[i],
                     "group_size": len(groups[gid[i]]), "in_heldout_refs": p in refs})
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    (OUT / "groups.json").write_text(json.dumps(multi, ensure_ascii=False, indent=1))
    (OUT / "split_audit.json").write_text(json.dumps({"tests_sharing_group_with_training": bad, "rows": audit},
                                                     ensure_ascii=False, indent=1))

    # out-of-domain subjects: CLIP text -> training images
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        ie = []
        for i in range(0, len(train_idx), 32):
            x = cp(images=[imgs[j] for j in train_idx[i:i + 32]], return_tensors="pt")
            ie.append(torch.nn.functional.normalize(clip.get_image_features(**x), dim=-1))
        ie = torch.cat(ie)
        contents = [c for c in json.loads((ROOT / "outputs/contents/contents.json").read_text()) if c["set"] == "ood"]
        te = torch.nn.functional.normalize(clip.get_text_features(**cp(text=[f"a papercut of {c['subject']}" for c in contents],
                                                                            return_tensors="pt", padding=True)), dim=-1)
    sims = (te @ ie.T).numpy()
    nn = {}
    sheet = Image.new("RGB", (W * 6, (W + 16) * len(contents)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (c, s) in enumerate(zip(contents, sims)):
        top = np.argsort(-s)[:5]
        nn[c["subject"]] = [(cat[train_idx[j]], round(float(s[j]), 3)) for j in top]
        la = Image.open(ROOT / f"outputs/contents/{c['id']}.png").convert("RGB")
        sheet.paste(la.resize((W, W)), (0, r * (W + 16) + 16))
        dr.text((3, r * (W + 16) + 2), c["subject"], fill="black")
        for k, j in enumerate(top):
            sheet.paste(imgs[train_idx[j]].resize((W, W)), ((k + 1) * W, r * (W + 16) + 16))
            dr.text(((k + 1) * W + 3, r * (W + 16) + 2), f"{Path(cat[train_idx[j]]).parent.name} {s[j]:.3f}", fill="black")
    sheet.save(OUT / "ood_subject_nn.jpg", quality=85)
    (OUT / "ood_subject_nn.json").write_text(json.dumps(nn, ensure_ascii=False, indent=1))

    print("entries", len(cat), "groups", len(groups), "multi-member groups", len(multi),
          "entries in multi groups", sum(len(v) for v in multi.values()))
    print("tests sharing a group with a training image:", bad)
    print("held-out refs (deduplicated):", len(refs))
    print(inv.groupby("role").size().to_dict())


if __name__ == "__main__":
    main()
