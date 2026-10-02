"""Extra image assets for the overview figure (paper/figures/tikz/fig_overview.tex).

- ov_work/labels/pieces/fine/coarse.jpg: the steps from the overview work to its cut lines.
- ov_canny.jpg: Canny edges of the overview work (the same map used to train the Canny-edge LoRA).
- ov_canny_vs.jpg: fine cut lines vs Canny on that work (black = both, red = Canny only, blue = cut line only).
- ov_svg.png: the enforced SVG plan of the rooster design, re-rasterised with piece outlines.
- q_sheet{0,1,2}.jpg: the three paper colours with most area in the cutting plan of the rooster design,
  each shown as the pieces cut from its own sheet (silhouette of the plan in light grey for orientation).
- ev_struct/proj.jpg, ev_mix.png: panel d, the measures on the rooster design (ours, seed 0): skeleton of the
  drawing found (black) or missed (red) by the design's edges; the design on the paper stock; the paper-colour
  shares q of the design and q_w of its nearest corpus work (same colour order).
"""
from pathlib import Path

import numpy as np
from PIL import Image

import cutlines
import priors
from skimage import color
from fig_cutline_process import diff_image, lines

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "paper/figures/assets"
REP = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
WORK = "plants/9.jpg"  # also shown in the cut-line figure, so no further work is reproduced
STEM = "data__整理的数据-常用_副本__库淑兰-data-处理后__Representative elements__" + WORK.replace("/", "__")
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")


def save(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(ASSET / name, quality=92)


def main():
    # the four steps from a work to its cut lines
    w = priors.load_rgb(REP / WORK, 512)
    save(w, "ov_work.jpg")
    pl = PAL.lab_with_white
    raw, _ = priors.nearest_palette(priors.to_lab(w), pl)
    rgbpal = np.clip(color.lab2rgb(pl[None])[0], 0, 1)
    save(rgbpal[raw], "ov_labels.jpg")
    proj, _, _, _ = priors.cut_project(w, PAL)
    save(proj, "ov_pieces.jpg")
    maps = cutlines.cutline_maps(REP / WORK, PAL, res=512)
    maps["dense"].convert("RGB").save(ASSET / "ov_fine.jpg", quality=92)
    maps["coarse"].convert("RGB").save(ASSET / "ov_coarse.jpg", quality=92)
    cl = ROOT / "outputs/cutlines"
    dense, canny = lines(cl / f"{STEM}__dense.png"), lines(cl / f"{STEM}__canny.png")
    Image.fromarray(np.where(canny, 0, 255).astype(np.uint8)).convert("RGB").save(ASSET / "ov_canny.jpg", quality=92)
    Image.fromarray((diff_image(dense, canny) * 255).astype(np.uint8)).save(ASSET / "ov_canny_vs.jpg", quality=92)

    plan = np.asarray(Image.open(ASSET / "q_rooster_plan.jpg").convert("RGB")).astype(np.float32) / 255
    rgb = np.vstack([PAL.rgb, [[1.0, 1.0, 1.0]]])
    lab = np.argmin(((plan[..., None, :] - rgb[None, None]) ** 2).sum(-1), -1)
    white = len(rgb) - 1
    counts = np.bincount(lab.ravel(), minlength=len(rgb))
    counts[white] = 0
    fg = lab != white
    for i, k in enumerate(np.argsort(counts)[::-1][:3]):
        out = np.ones_like(plan)
        out[fg] = 0.93
        out[lab == k] = rgb[k]
        Image.fromarray((out * 255).astype(np.uint8)).save(ASSET / f"q_sheet{i}.jpg", quality=92)
    print("assets written")



def svg_preview():
    """ov_svg.png: the enforced SVG plan of the rooster design, re-rasterised, with its piece outlines."""
    from scipy import ndimage as ndi
    from svg_validate import rasterise
    ras = rasterise(ROOT / "outputs/svg_enforced/qwen_q1/ood_rooster_s_seed0.svg").astype(np.float32) / 255
    if ras.ndim == 2:
        ras = np.repeat(ras[..., None], 3, -1)
    ras = ras[..., :3]
    key = (ras * 255).astype(np.int32) @ np.array([65536, 256, 1])
    edge = (key != ndi.shift(key, (1, 0), order=0, mode="nearest")) | (key != ndi.shift(key, (0, 1), order=0, mode="nearest"))
    out = 0.55 * ras + 0.45
    out[edge] = 0.1
    im = Image.fromarray((out * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)
    im.save(ASSET / "ov_svg.png")


def eval_assets():
    """ev_*: panel d, computed with the evaluation code itself (prints R, IoU and the colour-mix distance)."""
    from scipy import ndimage as ndi
    from scipy.spatial.distance import jensenshannon
    from skimage import morphology
    import palette_usage as pu
    import revision_metrics as rm
    rgb = priors.load_rgb(ROOT / "outputs/gen/qwen_q1/ood_rooster_s_seed0.png", 512)
    lns, sil, ground = rm.drawing("ood_rooster")
    st = rm.structure(rgb, lns, sil)
    gt = morphology.skeletonize(lns)
    near = ndi.distance_transform_edt(~priors.edge_map(rgb)) <= 3
    img = np.repeat(1 - 0.22 * (1 - rgb.mean(-1, keepdims=True)), 3, -1)
    img[ndi.binary_dilation(gt & near, morphology.disk(1))] = (0.10, 0.10, 0.12)
    img[ndi.binary_dilation(gt & ~near, morphology.disk(2))] = (0.85, 0.12, 0.10)
    save(img, "ev_struct.jpg")
    proj, labels, _, _ = priors.cut_project(rgb, PAL, ground=ground)
    save(proj, "ev_proj.jpg")
    q, H = pu.shares(labels), pu.corpus_hists()
    js = np.array([jensenshannon(q, h, base=2) for h in H])
    rgbpal = np.clip(color.lab2rgb(PAL.lab_with_white[None])[0], 0, 1)
    W, h, gap = 600, 46, 22
    bar = np.ones((2 * h + gap, W, 3))
    order = np.argsort(-q)
    for row, s in enumerate((q, H[js.argmin()])):
        x, y = 0, row * (h + gap)
        for c in order:
            n = int(round(s[c] * W))
            bar[y:y + h, x:x + n] = rgbpal[c]
            x += n
        bar[y:y + h, x:] = rgbpal[order[-1]] if x < W else 1
    Image.fromarray((bar * 255).astype(np.uint8)).save(ASSET / "ev_mix.png")
    print(f"rooster: R={st['line_recall_t3']:.3f} IoU={st['sil_iou']:.3f} D_mix={js.min():.3f}")


if __name__ == "__main__":
    main()
    svg_preview()
    eval_assets()
