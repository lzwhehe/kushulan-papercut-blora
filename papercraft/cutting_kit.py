"""Printable, numbered cutting kit for the fabrication study (review #01, Supplementary S6).

For a rule-based selection of enforced 200 x 200 mm plans, writes one A4 PDF per design:
  page 1   assembly view at 1:1 with piece numbers (assembly order: darkest sheet first)
  page 2.. one page per paper colour, piece outlines at 1:1 with numbers, for tracing/cutting
  last     recording sheet (paper used, time, modified/dropped/torn pieces, comments)
plus a CSV piece list (number, sheet colour, area in mm^2, centroid in mm).

Selection rule: CutCraft seed-0 plans sorted by piece count, six at evenly spaced quantiles;
the B-LoRA (animal reference) and CutCraft-Qwen plans of the first three of these drawings for comparison.
Print at 100 % (no "fit to page"); the 50 mm bar on each page checks the scale.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage import measure

import priors
from svg_validate import PX_MM, RES, SIZE_MM, rasterise

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "outputs/svg_enforced"
OUT = ROOT / "outputs/cutting_kit"
A4 = (210 / 25.4, 297 / 25.4)
LABEL_MIN_MM2 = 25.0  # smaller pieces are numbered in the list only (their number would not fit)


def plan_axes(fig):
    # 200 mm square placed 5 mm from the left and 40 mm from the top of an A4 page
    return fig.add_axes([5 / 210, (297 - 40 - SIZE_MM) / 297, SIZE_MM / 210, SIZE_MM / 297])


def scale_bar(fig):
    ax = fig.add_axes([5 / 210, 12 / 297, 50 / 210, 4 / 297])
    ax.plot([0, 50], [0, 0], "k-", lw=2)
    ax.set_xlim(0, 50); ax.axis("off")
    fig.text(57 / 210, 13 / 297, "50 mm (print at 100 %)", fontsize=7)


def pieces(labels, white):
    rows, masks = [], []
    order = [k for k in np.argsort(PAL_LAB[:, 0]) if k != white]
    n = 0
    for k in order:
        cc, m = ndi.label(labels == k)  # 4-connectivity, as in the projection
        props = sorted(measure.regionprops(cc), key=lambda p: (p.centroid[0], p.centroid[1]))
        for p in props:
            n += 1
            rows.append({"piece": n, "sheet": int(k), "area_mm2": p.area * PX_MM ** 2,
                         "cx_mm": p.centroid[1] * PX_MM, "cy_mm": p.centroid[0] * PX_MM})
            masks.append((n, k, cc == p.label))
    return pd.DataFrame(rows), masks


def design(svg_path, title, pdf_path):
    ras = rasterise(svg_path).astype(np.float32) / 255
    labels, _ = priors.nearest_palette(priors.to_lab(ras), PAL_LAB)
    white = len(PAL_LAB) - 1
    df, masks = pieces(labels, white)
    ext = [0, SIZE_MM, SIZE_MM, 0]
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=A4)
        fig.text(0.03, 0.97, f"{title} - assembly view (1:1, 200 x 200 mm)", fontsize=10, weight="bold", va="top")
        fig.text(0.03, 0.945, f"{len(df)} pieces on {df.sheet.nunique()} paper colours; paste in number order "
                              "(darkest sheet first). Machine-generated design, not a work by Ku Shulan.", fontsize=7, va="top")
        ax = plan_axes(fig)
        ax.imshow(PAL_RGB[labels], extent=ext, interpolation="nearest")
        for r in df.itertuples():
            if r.area_mm2 >= LABEL_MIN_MM2:
                ax.text(r.cx_mm, r.cy_mm, str(r.piece), fontsize=3.2, ha="center", va="center",
                        color="white" if PAL_LAB[r.sheet, 0] < 50 else "black")
        ax.set_xlim(0, SIZE_MM); ax.set_ylim(SIZE_MM, 0); ax.axis("off")
        scale_bar(fig); pdf.savefig(fig); plt.close(fig)
        for k in df.sheet.unique():
            fig = plt.figure(figsize=A4)
            sub = df[df.sheet == k]
            L = PAL_LAB[k]
            fig.text(0.03, 0.97, f"{title} - sheet {k}: {len(sub)} pieces", fontsize=10, weight="bold", va="top")
            fig.text(0.03, 0.945, f"paper colour CIELAB ({L[0]:.0f}, {L[1]:.0f}, {L[2]:.0f}); outlines at 1:1, "
                                  "as seen from the front (mirror when tracing on the back of the paper)", fontsize=7, va="top")
            sw = fig.add_axes([0.80, 0.935, 0.12, 0.035]); sw.imshow(PAL_RGB[k][None, None]); sw.axis("off")
            ax = plan_axes(fig)
            ax.imshow(np.ones((8, 8, 3)), extent=ext)
            for n, kk, m in masks:
                if kk != k:
                    continue
                for c in measure.find_contours(np.pad(m, 1).astype(float), 0.5):
                    ax.plot((c[:, 1] - 1) * PX_MM, (c[:, 0] - 1) * PX_MM, "k-", lw=0.25)
                r = df.iloc[n - 1]
                ax.text(r.cx_mm, r.cy_mm, str(n), fontsize=3.2 if r.area_mm2 >= LABEL_MIN_MM2 else 2.0, ha="center", va="center")
            ax.set_xlim(0, SIZE_MM); ax.set_ylim(SIZE_MM, 0); ax.axis("off")
            scale_bar(fig); pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=A4)
        fig.text(0.05, 0.95, f"{title} - fabrication record", fontsize=11, weight="bold")
        fields = ["Cutter (code)", "Date", "Paper used per sheet (brand, colour, weight)", "Tools", "Total time (min)",
                  "Pieces modified (numbers)", "Pieces merged (numbers)", "Pieces dropped (numbers)",
                  "Pieces torn / re-cut (numbers)", "Pieces that could not be placed", "Photograph file names",
                  "Comments (difficulty, fidelity to the plan, suggestions)"]
        for i, f in enumerate(fields):
            y = 0.9 - i * 0.07
            fig.text(0.05, y, f, fontsize=9)
            fig.add_artist(plt.Line2D([0.05, 0.95], [y - 0.03, y - 0.03], color="0.6", lw=0.5))
        pdf.savefig(fig); plt.close(fig)
    return df


def main():
    global PAL_LAB, PAL_RGB
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    PAL_LAB = pal.lab_with_white
    PAL_RGB = np.clip(priors.color.lab2rgb(PAL_LAB[None])[0], 0, 1)
    val = pd.read_csv(ROOT / "outputs/svg_validation_enforced.csv")
    cc = val[val.method == "cutcraft"].sort_values("pieces").reset_index(drop=True)
    pick = cc.iloc[np.linspace(0, len(cc) - 1, 6).round().astype(int)]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, r in enumerate(pick.itertuples()):
        cid = r.file.split("_s_seed")[0]
        jobs = [("cutcraft", SVG / "cutcraft" / (Path(r.file).stem + ".svg"))]
        if i < 3:
            jobs.append(("blora", SVG / "blora" / f"{cid}_s0_seed0.svg"))
            jobs.append(("qwen_q1", SVG / "qwen_q1" / f"{cid}_s_seed0.svg"))  # CutCraft-Qwen, same drawing
        for m, svg in jobs:
            code = f"K{len(manifest) + 1:02d}"  # neutral code, method revealed only in the manifest
            df = design(svg, f"Design {code}", OUT / f"{code}.pdf")
            df.to_csv(OUT / f"{code}_pieces.csv", index=False)
            manifest.append({"code": code, "method": m, "drawing": cid, "svg": str(svg.relative_to(ROOT)),
                             "pieces": len(df), "sheets": int(df.sheet.nunique()), "smallest_mm2": round(df.area_mm2.min(), 2)})
            print(manifest[-1])
    (OUT / "manifest_UNBLIND.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
