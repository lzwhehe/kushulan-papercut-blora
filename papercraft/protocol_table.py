"""Cut-projection protocol analysis (Table 5) and per-category breakdown (Table 7)."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
SHEET_MM2 = 200 * 200  # square design exported at 200 x 200 mm (Methods)
SETTINGS = [("no filtering, no ground", 0.0, False), ("$\\alpha=5{\\times}10^{-5}$", 5e-5, True),
            ("$\\alpha=2{\\times}10^{-4}$ (default)", 2e-4, True), ("$\\alpha=10^{-3}$", 1e-3, True),
            ("$\\alpha=5{\\times}10^{-3}$", 5e-3, True), ("default without ground", 2e-4, False)]


def one(args):
    path, cid = args
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    rgb = priors.load_rgb(path, 512)
    la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
    lines = la.mean(-1) < 0.85
    sil = priors.silhouette_from_lineart(lines)
    ground = priors.ground_from_lineart(lines)
    rows = []
    for name, alpha, use_g in SETTINGS:
        if alpha == 0:
            lab = priors.to_lab(rgb)
            labels, _ = priors.nearest_palette(lab, pal.lab_with_white)
            proj = np.clip(priors.color.lab2rgb(pal.lab_with_white[labels][None])[0], 0, 1)
        else:
            proj, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=alpha, ground=ground if use_g else None)
        white = len(pal.lab_with_white) - 1
        sheets, pieces, smallest = 0, 0, np.inf
        for k in np.unique(labels):
            if k == white:
                continue
            m = labels == k
            if m.mean() > 1e-3:
                sheets += 1
            cc, n = ndi.label(m)
            if n:
                areas = np.bincount(cc.ravel())[1:]
                pieces += n
                smallest = min(smallest, areas.min())
        e = priors.edge_f1(priors.edge_map(proj.astype(np.float32)), lines, 3)  # keys edge_p, edge_r, edge_f1
        rows.append({"setting": name, "sheets": sheets, "pieces": pieces,
                     "smallest_mm2": smallest / labels.size * SHEET_MM2,
                     "edge_f1": e["edge_f1"], "recall": e["edge_r"], "prec": e["edge_p"], "sil_iou": priors.silhouette_iou(proj.astype(np.float32), sil),
                     "change_de": float(priors.color.deltaE_ciede2000(priors.to_lab(rgb), priors.to_lab(proj)).mean())})
    return rows


def protocol():
    files = sorted((G / "cutcraft").glob("*_seed*.png"))
    args = [(str(f), f.name.split("_s_seed")[0]) for f in files]
    with ProcessPoolExecutor(24) as ex:
        rows = [r for rs in ex.map(one, args) for r in rs]
    df = pd.DataFrame(rows)
    t = df.groupby("setting", sort=False).agg(sheets=("sheets", "mean"), pieces=("pieces", "median"),
                                             smallest=("smallest_mm2", "median"), edge_f1=("edge_f1", "mean"),
                                             recall=("recall", "mean"), prec=("prec", "mean"),
                                             sil_iou=("sil_iou", "mean"), change=("change_de", "mean"))
    t.to_csv(ROOT / "outputs/table_protocol.csv")
    lines = [f"{n} & {r.sheets:.1f} & {r.pieces:.0f} & {r.smallest:.2f} & {r.change:.1f} & {r.recall:.2f} & {r.prec:.2f} & {r.sil_iou:.2f} \\\\"
             for n, r in t.iterrows()]
    (ROOT / "paper/sections/table_protocol.tex").write_text("\n".join(lines) + "\n")
    print(t.round(3).to_string())


def per_category():
    contents = {c["id"]: c for c in json.loads((ROOT / "outputs/contents/contents.json").read_text())}
    ood = json.loads((ROOT / "outputs/audit/ood_subject_review.json").read_text())

    def cat(cid):
        c = contents[cid]
        if c["set"] == "indomain":
            return {"person": "Figures", "animal": "Animals", "plant": "Plants", "daily object": "Daily objects",
                    "window flower": "Window flowers", "border": "Borders"}[c["category"]]
        if c["set"] == "repo":
            return "Repository drawings"
        subj = c.get("subject") or c.get("prompt") or ""
        if any(s in subj for s in ood["not_found"]):
            return "New: no related work found"
        if any(s in subj for s in ood["partially_represented"]):
            return "New: partly represented"
        return "New: represented in corpus"
    order = ["Figures", "Animals", "Plants", "Daily objects", "Window flowers", "Borders", "Repository drawings",
             "New: represented in corpus", "New: partly represented", "New: no related work found"]
    out = []
    for m, lab in (("blora", "B-LoRA"), ("cutcraft", "CutCraft")):
        raw = pd.read_csv(G / m / "metrics.csv")[["file", "content", "palette_de"]].merge(
            pd.read_csv(G / m / "rev_metrics.csv").drop(columns=["content"]), on="file")
        raw["cat"] = raw.content.map(cat)
        d = raw.groupby(["cat", "content"])[["palette_de", "raw_line_recall_t3", "clip_style_dedup", "proj_line_recall_t3",
                                             "proj_sil_iou"]].mean().reset_index()
        t = d.groupby("cat")[["palette_de", "raw_line_recall_t3", "clip_style_dedup", "proj_line_recall_t3", "proj_sil_iou"]].mean()
        t["n"] = d.groupby("cat").content.nunique()
        t = t.reindex(order)
        t["method"] = lab
        out.append(t)
    df = pd.concat(out)
    df.to_csv(ROOT / "outputs/table_category.csv")
    bl, cc = out
    lines = []
    for c in order:
        r0, r1 = bl.loc[c], cc.loc[c]
        lines.append(f"{c} & {int(r1.n)} & {r0.palette_de:.1f} / {r1.palette_de:.1f} & {r0.raw_line_recall_t3:.2f} / {r1.raw_line_recall_t3:.2f} & "
                     f"{r0.clip_style_dedup:.3f} / {r1.clip_style_dedup:.3f} & {r1.proj_line_recall_t3:.2f} & {r1.proj_sil_iou:.2f} \\\\")
    (ROOT / "paper/sections/table_category.tex").write_text("\n".join(lines) + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    per_category()
    protocol()
