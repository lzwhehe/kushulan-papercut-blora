"""Craft priors for Ku Shulan papercut: palette, papercut metrics and cut projection.

Everything here is plain NumPy / scikit-image so that metrics are independent of
the generative model that produced an image. Differentiable counterparts used for
training and guidance live in ``energies.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, feature, morphology

WHITE_L, WHITE_C = 90.0, 10.0  # background: very light and nearly achromatic


# ----------------------------------------------------------------------------- io
def load_rgb(path, size: int | None = 512) -> np.ndarray:
    """Load an image as float RGB in [0,1], compositing transparency on white."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if size is not None:
        im = pad_to_square(im).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def pad_to_square(im: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return color.rgb2lab(np.clip(rgb, 0, 1))


def foreground_mask(lab: np.ndarray) -> np.ndarray:
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return ~((lab[..., 0] > WHITE_L) & (chroma < WHITE_C))


# ------------------------------------------------------------------------ palette
@dataclass
class Palette:
    lab: np.ndarray  # (K,3) foreground colours; white background is implicit
    weight: np.ndarray  # (K,) pixel share in the corpus

    @property
    def rgb(self) -> np.ndarray:
        return np.clip(color.lab2rgb(self.lab[None])[0], 0, 1)

    @property
    def lab_with_white(self) -> np.ndarray:
        return np.concatenate([self.lab, [[100.0, 0.0, 0.0]]], 0)

    def save(self, path):
        Path(path).write_text(json.dumps({
            "lab": self.lab.round(3).tolist(),
            "weight": self.weight.round(5).tolist(),
            "hex": ["#%02x%02x%02x" % tuple((c * 255).round().astype(int)) for c in self.rgb],
        }, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["lab"], np.float32), np.array(d["weight"], np.float32))


def fit_palette(paths, k: int = 12, per_image: int = 4000, seed: int = 0) -> Palette:
    """K-means in CIELAB over foreground pixels, each image contributing equally."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    samples = []
    for p in paths:
        lab = to_lab(load_rgb(p, 384)).reshape(-1, 3)
        fg = lab[foreground_mask(lab.reshape(384, 384, 3)).reshape(-1)]
        if len(fg) == 0:
            continue
        samples.append(fg[rng.choice(len(fg), min(per_image, len(fg)), replace=False)])
    x = np.concatenate(samples)
    km = KMeans(k, n_init=8, random_state=seed).fit(x)
    w = np.bincount(km.labels_, minlength=k) / len(x)
    order = np.argsort(-w)
    return Palette(km.cluster_centers_[order].astype(np.float32), w[order].astype(np.float32))


def nearest_palette(lab: np.ndarray, pal_lab: np.ndarray):
    """Return (index, CIEDE2000 distance) of the nearest palette colour per pixel."""
    flat = lab.reshape(-1, 3)
    # pre-select with Euclidean Lab distance, then CIEDE2000 on the best 3 candidates
    d2 = ((flat[:, None, :] - pal_lab[None]) ** 2).sum(-1)
    cand = np.argsort(d2, 1)[:, :3]
    de = np.stack([color.deltaE_ciede2000(flat, pal_lab[cand[:, j]]) for j in range(3)], 1)
    j = de.argmin(1)
    idx = cand[np.arange(len(flat)), j]
    return idx.reshape(lab.shape[:-1]), de[np.arange(len(flat)), j].reshape(lab.shape[:-1])


# ----------------------------------------------------------------- cut projection
def cut_project(rgb: np.ndarray, pal: Palette, min_area_frac: float = 2e-4, smooth: int = 2):
    """Project an image onto a cuttable layered papercut.

    1. assign each pixel to the nearest palette colour (white = background);
    2. majority (mode) filter to remove single-pixel speckle;
    3. merge islands smaller than ``min_area_frac`` of the image into the
       surrounding label, i.e. pieces that are too small to cut by hand.
    Returns (projected_rgb, label_map, stats).
    """
    lab = to_lab(rgb)
    pal_lab = pal.lab_with_white
    labels, _ = nearest_palette(lab, pal_lab)
    raw_labels = labels.copy()
    if smooth:
        labels = _mode_filter(labels, len(pal_lab), smooth)
    h, w = labels.shape
    min_area = max(1, int(min_area_frac * h * w))
    n_small, first = 0, None
    for _ in range(3):
        small = np.zeros_like(labels, bool)
        for k in np.unique(labels):
            cc, n = ndi.label(labels == k)
            if n:
                areas = np.bincount(cc.ravel())
                areas[0] = min_area  # background of this mask is never "small"
                s = areas[cc] < min_area
                n_small += int((areas[1:] < min_area).sum())
                small |= s
        first = n_small if first is None else first
        if not small.any() or small.all():
            break
        # every sub-cuttable island takes the label of the nearest kept pixel
        _, (iy, ix) = ndi.distance_transform_edt(small, return_indices=True)
        labels = labels[iy, ix]
    out_rgb = np.clip(color.lab2rgb(pal_lab[labels][None])[0], 0, 1)
    stats = {"islands_merged": n_small, "islands_first_pass": first, "layers": int(len(np.unique(labels)))}
    return out_rgb.astype(np.float32), labels, raw_labels, stats


def _mode_filter(labels, k, r):
    fp = morphology.disk(r).astype(np.float32)
    votes = np.stack([ndi.convolve((labels == i).astype(np.float32), fp, mode="nearest") for i in range(k)])
    return votes.argmax(0)


def export_layers_svg(labels: np.ndarray, pal: Palette, path, simplify: float = 1.0):
    """Write one closed-path SVG group per paper colour (bottom = darkest)."""
    from skimage import measure

    pal_lab = pal.lab_with_white
    rgb = np.clip(color.lab2rgb(pal_lab[None])[0], 0, 1)
    h, w = labels.shape
    order = [k for k in np.argsort(pal_lab[:, 0]) if k != len(pal_lab) - 1]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for k in order:
        m = labels == k
        if not m.any():
            continue
        hexc = "#%02x%02x%02x" % tuple((rgb[k] * 255).round().astype(int))
        parts.append(f'<g id="layer{k}" fill="{hexc}" fill-rule="evenodd"><path d="')
        for c in measure.find_contours(np.pad(m, 1).astype(float), 0.5):
            c = measure.approximate_polygon(c - 1, simplify)
            if len(c) < 3:
                continue
            parts.append("M" + " L".join(f"{x:.1f},{y:.1f}" for y, x in c) + "Z ")
        parts.append('"/></g>')
    parts.append("</svg>")
    Path(path).write_text("".join(parts))


# ---------------------------------------------------------------- craft metrics
def colour_purity(lab: np.ndarray, fg: np.ndarray, k: int = 8, tau: float = 6.0, n: int = 20000) -> float:
    """Share of foreground pixels within ΔE00 < tau of the image's own k dominant colours.

    Cut paper is locally uniform, so its colour histogram is a few sharp spikes;
    shading, blur, gradients and texture all create in-between colours.
    """
    from sklearn.cluster import KMeans

    x = lab[fg]
    rng = np.random.default_rng(0)
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)]
    km = KMeans(min(k, len(x)), n_init=3, random_state=0).fit(x)
    de = color.deltaE_ciede2000(x, km.cluster_centers_[km.labels_])
    return float((de < tau).mean())



def craft_metrics(rgb: np.ndarray, pal: Palette, grad_tau: float = 6.0) -> dict:
    """Papercut-specific image statistics (all computed at the input resolution)."""
    lab = to_lab(rgb)
    fg = foreground_mask(lab)
    fg_frac = float(fg.mean())
    if fg.sum() < 50:
        return {"fg_frac": fg_frac}
    _, de = nearest_palette(lab, pal.lab_with_white)
    gy = np.stack([ndi.sobel(lab[..., c], 0) for c in range(3)], -1) / 8.0
    gx = np.stack([ndi.sobel(lab[..., c], 1) for c in range(3)], -1) / 8.0
    gmag = np.sqrt((gx ** 2 + gy ** 2).sum(-1))
    proj, labels, raw, st = cut_project(rgb, pal)
    de_proj = color.deltaE_ciede2000(lab, to_lab(proj))
    # interior deviation: distance of each pixel >=2 px inside a cut region to that
    # region's mean colour -> shading, gradients and texture that paper cannot show
    edge = np.zeros_like(labels, bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    interior = fg & ~ndi.binary_dilation(edge, morphology.disk(1))
    reg, off = np.zeros_like(labels), 0
    for kk in np.unique(labels):
        c, m = ndi.label(labels == kk)
        reg[c > 0] = c[c > 0] + off
        off += m
    cnt = np.bincount(reg.ravel(), minlength=off + 1).astype(np.float64)
    mean = np.stack([np.bincount(reg.ravel(), lab[..., c].ravel(), minlength=off + 1) for c in range(3)], -1)
    mean /= np.maximum(cnt, 1)[:, None]
    dev = color.deltaE_ciede2000(lab, mean[reg])
    return {
        "fg_frac": fg_frac,
        "purity": colour_purity(lab, fg),
        "palette_de": float(de[fg].mean()),  # ΔE00 to nearest Ku Shulan colour
        "flatness": float((gmag[fg] < grad_tau).mean()),  # share of low-gradient pixels
        "interior_de": float(dev[interior].mean()) if interior.any() else float("nan"),
        "cut_residual": float(de_proj[fg].mean()),  # ΔE00 change needed to become cuttable
        "fragments": st["islands_first_pass"],  # sub-cuttable islands after speckle filter
        "chroma": float(np.hypot(lab[..., 1], lab[..., 2])[fg].mean()),
    }


# ---------------------------------------------------------- structure metrics
def lineart_mask(rgb: np.ndarray, thresh: float = 0.6) -> np.ndarray:
    g = rgb.mean(-1)
    m = g < thresh
    return morphology.remove_small_objects(m, 8)


def silhouette_from_lineart(lines: np.ndarray, r: int = 3) -> np.ndarray:
    """Filled outer shape of a (possibly not perfectly closed) line drawing."""
    d = ndi.binary_dilation(lines, morphology.disk(r))
    return ndi.binary_erosion(ndi.binary_fill_holes(d), morphology.disk(r))


def edge_map(rgb: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    lab = to_lab(rgb)
    e = np.zeros(lab.shape[:2], bool)
    for c, s in ((0, 1.0), (1, 0.6), (2, 0.6)):
        e |= feature.canny(lab[..., c] * s, sigma=sigma, low_threshold=4, high_threshold=10)
    return e


def edge_f1(pred_edges: np.ndarray, gt_lines: np.ndarray, tol: int = 3) -> dict:
    gt = morphology.skeletonize(gt_lines)
    if gt.sum() == 0 or pred_edges.sum() == 0:
        return {"edge_p": 0.0, "edge_r": 0.0, "edge_f1": 0.0}
    dt_gt = ndi.distance_transform_edt(~gt)
    dt_pr = ndi.distance_transform_edt(~pred_edges)
    p = float((dt_gt[pred_edges] <= tol).mean())
    r = float((dt_pr[gt] <= tol).mean())
    return {"edge_p": p, "edge_r": r, "edge_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


def silhouette_iou(rgb: np.ndarray, gt_sil: np.ndarray) -> float:
    fg = foreground_mask(to_lab(rgb))
    fg = ndi.binary_fill_holes(morphology.binary_closing(fg, morphology.disk(2)))
    inter = (fg & gt_sil).sum()
    union = (fg | gt_sil).sum()
    return float(inter / max(union, 1))


def symmetry_score(rgb: np.ndarray) -> float:
    """1 - mean ΔE between the image and its left–right mirror, normalised to [0,1]."""
    lab = to_lab(rgb)
    fg = foreground_mask(lab) | foreground_mask(lab[:, ::-1])
    d = np.linalg.norm(lab - lab[:, ::-1], axis=-1)
    return float(1 - np.clip(d[fg].mean() / 100.0, 0, 1))
