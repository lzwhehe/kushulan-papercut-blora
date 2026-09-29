"""Review #32-#34: physical-scale SVG export and validation of the delivered cutting plans.

Every design is exported at an explicit physical size (200 x 200 mm square; 1024-px label map,
so 1 px = 0.195 mm) with a minimum piece area of 8 mm^2 (alpha = 8 / 200^2 = 2e-4). The exported
SVG is then re-rasterised and checked:
  pieces_below_min   connected single-colour pieces smaller than 8 mm^2 in the re-rasterised file
  thin_area_share    share of piece area narrower than 1 mm (removed by an opening of radius 0.5 mm)
  pieces_with_thin_strip  pieces whose sub-1-mm part exceeds 2 mm^2 (a strip, not a corner)
  holes              interior holes (enclosed regions of other colours) summed over pieces
  invalid_paths      closed sub-paths that are not valid simple polygons (shapely)
  raster_agreement   share of pixels whose colour in the re-rasterised SVG equals the label map
The plan is a colour-separated mosaic: pieces of different colours abut and do not overlap.
An optional backing sheet (figure silhouette dilated by 2 mm) can be written as the bottom layer.
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology

import priors

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "outputs/gen"
SIZE_MM, RES, MIN_MM2, NECK_MM = 200.0, 1024, 8.0, 1.0
PX_MM = SIZE_MM / RES


def export_svg(labels, pal, path, backing=False, backing_mm=2.0, simplify=1.0):
    pal_lab = pal.lab_with_white
    white = len(pal_lab) - 1
    rgb = np.clip(priors.color.lab2rgb(pal_lab[None])[0], 0, 1)
    h, w = labels.shape
    hexc = lambda k: "#%02x%02x%02x" % tuple((rgb[k] * 255).round().astype(int))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE_MM:g}mm" height="{SIZE_MM:g}mm" viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="#ffffff"/>']

    def group(mask, colour, gid):
        s = [f'<g id="{gid}" fill="{colour}" fill-rule="evenodd"><path d="']
        for c in measure.find_contours(np.pad(mask, 1).astype(float), 0.5):
            c = measure.approximate_polygon(c - 1, simplify)
            if len(c) >= 3:
                s.append("M" + " L".join(f"{x:.1f},{y:.1f}" for y, x in c) + "Z ")
        s.append('"/></g>')
        return "".join(s)

    if backing:
        fig = ndi.binary_dilation(labels != white, morphology.disk(max(1, int(round(backing_mm / PX_MM)))))
        parts.append(group(fig, "#1a1a1a", "backing_sheet"))
    for k in [k for k in np.argsort(pal_lab[:, 0]) if k != white]:
        m = labels == k
        if m.any():
            parts.append(group(m, hexc(k), f"sheet_{k}"))
    parts.append("</svg>")
    Path(path).write_text("".join(parts))


def rasterise(svg_path):
    import pymupdf
    pymupdf.TOOLS.set_aa_level(0)  # no anti-aliasing: blended edge pixels would create spurious pieces
    doc = pymupdf.open(str(svg_path))
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(RES / doc[0].rect.width, RES / doc[0].rect.height), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:RES, :RES]


def path_validity(svg_path):
    from shapely.geometry import Polygon
    txt = Path(svg_path).read_text()
    bad = total = 0
    for sub in re.findall(r"M([^Z]+)Z", txt):
        pts = [tuple(map(float, p.split(","))) for p in sub.replace("L", " ").split()]
        if len(pts) < 3:
            continue
        total += 1
        bad += not Polygon(pts).is_valid
    return bad, total


ENFORCE = False


def validate(args):
    img_path, cid, out_dir, enforce = args
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    pal_lab = pal.lab_with_white
    white = len(pal_lab) - 1
    rgb = priors.load_rgb(img_path, RES)
    la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
    g = priors.ground_from_lineart(la.mean(-1) < 0.85)
    g = np.asarray(Image.fromarray(g.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 0
    # enforcement uses the same opening radius as the validation below, and a 10 % area safety margin
    width_px = 2 * max(1, int(round(NECK_MM / 2 / PX_MM))) if enforce else 0
    area_frac = MIN_MM2 / SIZE_MM ** 2 * (1.1 if enforce else 1.0)
    _, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=area_frac, ground=g,
                                         min_width_px=width_px, enforce=enforce)
    svg = Path(out_dir) / (Path(img_path).stem + ".svg")
    export_svg(labels, pal, svg, simplify=0.0 if enforce else 1.0)
    # structure retained by the delivered plan (line recall against the drawing, at 512 px)
    small = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    plan = np.clip(priors.color.lab2rgb(pal_lab[small][None])[0], 0, 1).astype(np.float32)
    lines = la.mean(-1) < 0.85
    recall = priors.edge_f1(priors.edge_map(plan), lines, 3)["edge_r"]
    ras = rasterise(svg).astype(np.float32) / 255
    rl, _ = priors.nearest_palette(priors.to_lab(ras), pal_lab)
    agree = float((rl == labels).mean())
    min_px = MIN_MM2 / PX_MM ** 2
    neck_r = max(1, int(round(NECK_MM / 2 / PX_MM)))
    below = neck = holes = pieces = 0
    thin_px = area_px = 0
    smallest = np.inf
    for k in np.unique(rl):
        if k == white:
            continue
        cc, n = ndi.label(rl == k)
        for i in range(1, n + 1):
            m = cc == i
            a = m.sum()
            pieces += 1
            smallest = min(smallest, a * PX_MM ** 2)
            below += a < min_px
            resid = (m & ~morphology.binary_opening(m, morphology.disk(neck_r))).sum()
            thin_px += resid
            area_px += a
            if resid * PX_MM ** 2 > 2.0:  # a thin strip (< 1 mm wide) of more than 2 mm^2, not just a corner
                neck += 1
            holes += max(0, ndi.label(~m)[1] - 1)
    bad, total = path_validity(svg)
    return {"file": Path(img_path).name, "pieces": pieces, "pieces_below_min": int(below),
            "pieces_with_thin_strip": int(neck), "thin_area_share": float(thin_px / max(area_px, 1)), "holes": int(holes), "smallest_mm2": float(smallest),
            "invalid_paths": bad, "paths": total, "raster_agreement": agree,
            "svg_kb": svg.stat().st_size / 1024, "plan_line_recall": recall}


def main(methods, enforce=False):
    rows = []
    for m in methods:
        out_dir = ROOT / f"outputs/svg{'_enforced' if enforce else ''}/{m}"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((G / m).glob("*_seed0.png"))
        args = [(str(f), f.name.split("_s")[0], str(out_dir), enforce) for f in files]
        with ProcessPoolExecutor(20) as ex:
            res = list(ex.map(validate, args))
        df = pd.DataFrame(res)
        df["method"] = m
        rows.append(df)
        print(m, len(df), "designs with pieces<8mm2: %d/%d" % ((df.pieces_below_min > 0).sum(), len(df)),
              "median pieces %d" % df.pieces.median(), "thin strips/design %.1f" % df.pieces_with_thin_strip.median(),
              "thin area %.3f" % df.thin_area_share.mean(), "pieces<8mm2 per design %.2f" % df.pieces_below_min.mean(),
              "invalid paths %d/%d" % (df.invalid_paths.sum(), df.paths.sum()), "raster agreement %.4f" % df.raster_agreement.mean(),
              "plan line recall %.3f" % df.plan_line_recall.mean())
    out = ROOT / f"outputs/svg_validation{'_enforced' if enforce else ''}.csv"
    new = pd.concat(rows)
    if out.exists():  # keep rows of methods not re-run
        old = pd.read_csv(out)
        new = pd.concat([old[~old.method.isin(methods)], new])
    new.to_csv(out, index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    enf = "--enforce" in a
    main([x for x in a if x != "--enforce"] or ["cutcraft"], enforce=enf)
