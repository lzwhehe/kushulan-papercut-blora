"""Figures in the layout of the npj Heritage Science reference article (Figs. 1, 7, 8, 9)
and image assets for the TikZ method diagrams (Figs. 2-6)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import priors

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
ASSET = FIG / "assets"
G = ROOT / "outputs/gen"
PAL = priors.Palette.load(ROOT / "outputs/data/palette.json")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6, "savefig.dpi": 300})
NAVY, RED, LBLUE, SALMON, GOLD = "#2d5277", "#a3302f", "#8fb3d1", "#e08a7e", "#e6b35a"


def im(path, size=512):
    return Image.fromarray((priors.load_rgb(path, size) * 255).astype(np.uint8))


def gen(method, cid, seed=0, style=None):
    s = "" if style is None else str(style)
    return G / method / f"{cid}_s{s}_seed{seed}.png"


def ground_of(cid):
    la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
    return priors.ground_from_lineart(la.mean(-1) < 0.85)


def projected(path, cid, size=512):
    rgb = priors.load_rgb(path, size)
    g = ground_of(cid)
    if size != 512:
        g = np.asarray(Image.fromarray(g.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0
    return Image.fromarray((priors.cut_project(rgb, PAL, ground=g)[0] * 255).astype(np.uint8))


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03, **({"pil_kwargs": {"quality": 92}} if name.endswith(".jpg") else {}))
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    rep = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
    a = im(rep / "animals/2.jpg", 512)  # layered colour pieces over a dark ground
    b = im(rep / "person/1.jpg", 1024).crop((256, 300, 768, 812))  # dense dots, crescents and saw-teeth
    c = Image.open(gen("prompt_cn", "ood_rooster")).convert("RGB")  # generic generation: shading, pastel, 3-D paper
    d = Image.open(gen("blora", "ood_teapot", 0, 0)).convert("RGB")  # per-drawing B-LoRA: repeated structure
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, x, lab in zip(axes, [a, b, c, d], "abcd"):
        ax.imshow(x.resize((512, 512)))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"({lab})", fontsize=15, fontweight="bold", labelpad=6)
    plt.subplots_adjust(wspace=0.04)
    save(fig, "fig1_challenges.jpg")


# ---------------------------------------------------------------- Fig. 7
def fig7():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    import glob
    rep = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
    cats = [("Animals", "animals"), ("Plants", "plants"), ("Figures", "person"), ("Daily objects", "daily"),
            ("Window flowers", "window decoration"), ("Borders", "frame")]
    counts = [len(glob.glob(str(rep / c / "*"))) for _, c in cats]
    npat = len(glob.glob(str(ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Pattern_symbols/*/*")))
    labels = [n for n, _ in cats] + ["Pattern symbols"]
    vals = counts + [npat]
    cols = [NAVY, "#3f7a4f", RED, GOLD, "#7a5aa6", "#4f97a3", "#b9b3a6"]
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_axes([0.0, 0.08, 0.27, 0.84])
    w, _, autot = ax.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.32, edgecolor="white"),
                         autopct=lambda p: f"{p:.1f}%" if p > 6 else "", pctdistance=0.84,
                         textprops=dict(color="white", fontsize=8.5, fontweight="bold"))
    ax.text(0, 0.12, "Total", ha="center", fontsize=13)
    ax.text(0, -0.13, f"{sum(vals)}", ha="center", fontsize=24)
    ax.text(0, -0.36, "works", ha="center", fontsize=13)
    ax.legend(w, [f"{l} ({v})" for l, v in zip(labels, vals)], loc="center left", bbox_to_anchor=(0.97, 0.5),
              frameon=False, fontsize=8.5)
    ax.set_title("Ku Shulan catalogue", fontsize=12, pad=4)
    bx = fig.add_axes([0.66, 0.12, 0.33, 0.8])
    rows = [("Training works\n(style block)", len(split["train"]), NAVY),
            ("Held-out duplicates\n(never trained)", len(split["held_out"]), LBLUE),
            ("In-domain outlines\n(test, with original)", len(split["test_indomain"]), RED),
            ("Out-of-domain drawings\n(test)", 12, SALMON),
            ("Repository drawings\n(test)", 3, GOLD),
            ("Development drawings\n(tuning only)", 3, "#9a9a9a")]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, c) in zip(y, rows):
        bx.barh(yy, v, height=0.16, color=c)
        bx.text(v + 3, yy, str(v), va="center", color=c, fontsize=10)
    bx.set_yticks(y, [r[0] for r in rows], fontsize=10)
    bx.set_xlim(0, 280)
    bx.grid(axis="x", color="#e6e6e6", lw=0.8)
    bx.set_axisbelow(True)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Number of images", fontsize=12)
    save(fig, "fig7_dataset.pdf")




# ---------------------------------------------------------------- assets for TikZ diagrams
def _save_arr(a, name, size=256):
    x = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)) if a.dtype != np.uint8 else Image.fromarray(a)
    x.resize((size, size), Image.NEAREST if a.ndim == 2 else Image.LANCZOS).convert("RGB").save(ASSET / name, quality=92)


def _cmap(a, cmap="jet", vmin=0, vmax=None):
    import matplotlib.cm as cm
    vmax = vmax if vmax is not None else float(np.percentile(a, 99))
    return (matplotlib.colormaps[cmap](np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0, 1))[..., :3])


def assets():
    from scipy import ndimage as ndi
    from skimage import color, morphology
    import cutlines
    ASSET.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    rep = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Representative elements"
    # corpus thumbnails (training works only)
    for i, p in enumerate(["animals/2.jpg", "person/1.jpg", "plants/1.jpg", "window decoration/2.jpg",
                           "daily/1.jpg", "frame/1.jpg"]):
        im(rep / p, 256).save(ASSET / f"corpus{i}.jpg", quality=92)
    # palette strip
    fig, ax = plt.subplots(figsize=(4, 0.5))
    x = 0
    for c, w in zip(PAL.rgb, PAL.weight):
        ax.add_patch(plt.Rectangle((x, 0), w, 1, color=c, lw=0)); x += w
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    fig.savefig(ASSET / "palette.png", bbox_inches="tight", pad_inches=0, dpi=200); plt.close(fig)
    # craft-prior pipeline on one training work
    w = priors.load_rgb(rep / "animals/2.jpg", 512)
    Image.fromarray((w * 255).astype(np.uint8)).save(ASSET / "cp_work.jpg", quality=92)
    lab = priors.to_lab(w)
    pl = PAL.lab_with_white
    raw, de = priors.nearest_palette(lab, pl)
    rgbpal = np.clip(color.lab2rgb(pl[None])[0], 0, 1)
    _save_arr(rgbpal[raw], "cp_labels.jpg", 512)
    _save_arr(_cmap(de, "jet", 0, 20), "cp_de.jpg", 512)
    proj, labels, _, _ = priors.cut_project(w, PAL)
    _save_arr(proj, "cp_proj.jpg", 512)
    maps = cutlines.cutline_maps(rep / "animals/2.jpg", PAL, res=512)
    maps["dense"].convert("RGB").save(ASSET / "cp_dense.jpg", quality=92)
    maps["coarse"].convert("RGB").save(ASSET / "cp_coarse.jpg", quality=92)
    # control images as the ControlNet sees them (white lines on black)
    for k in ("dense", "coarse"):
        a = np.asarray(maps[k]) < 128
        Image.fromarray((a * 255).astype(np.uint8)).convert("RGB").save(ASSET / f"cp_ctrl_{k}.jpg", quality=92)
    # CraftGuide energy maps on a CutCraft output and its unguided counterpart
    for tag, meth in (("cl", "cutline"), ("cc", "cutcraft")):
        cid = "ood_rooster"
        x = priors.load_rgb(gen(meth, cid), 512)
        Image.fromarray((x * 255).astype(np.uint8)).save(ASSET / f"cg_{tag}_img.jpg", quality=92)
        lx = priors.to_lab(x)
        _, dx = priors.nearest_palette(lx, pl)
        _save_arr(_cmap(dx, "jet", 0, 20), f"cg_{tag}_pal.jpg", 512)
        gx = np.sqrt(sum(ndi.sobel(lx[..., c], 1) ** 2 + ndi.sobel(lx[..., c], 0) ** 2 for c in range(3))) / 8
        _save_arr(_cmap(1 - np.exp(-(gx ** 2) / 64), "magma", 0, 1), f"cg_{tag}_flat.jpg", 512)
        g = ground_of(cid)
        dwhite = np.linalg.norm(lx - np.array([100, 0, 0]), axis=-1) * g
        _save_arr(_cmap(dwhite, "jet", 0, 60), f"cg_{tag}_ground.jpg", 512)
    la = priors.load_rgb(ROOT / "outputs/contents/ood_rooster.png", 512)
    Image.fromarray((la * 255).astype(np.uint8)).save(ASSET / "cg_line.jpg", quality=92)
    _save_arr(np.where(ground_of("ood_rooster")[..., None], np.array([0.85, 0.9, 1.0]), np.array([1.0, 0.9, 0.75])),
              "cg_groundmask.jpg", 512)
    # cut projection steps on the CutCraft rooster
    x = priors.load_rgb(gen("cutcraft", "ood_rooster"), 512)
    lx = priors.to_lab(x)
    raw, _ = priors.nearest_palette(lx, pl)
    _save_arr(rgbpal[raw], "pj_nearest.jpg", 512)
    g = ground_of("ood_rooster")
    lg = raw.copy(); lg[g] = len(pl) - 1
    _save_arr(rgbpal[lg], "pj_ground.jpg", 512)
    mf = priors._mode_filter(lg, len(pl), 2)
    _save_arr(rgbpal[mf], "pj_mode.jpg", 512)
    proj, labels, _, st = priors.cut_project(x, PAL, ground=g)
    _save_arr(proj, "pj_final.jpg", 512)
    # highlight the islands that were merged (sub-cuttable pieces) in red over a grey copy
    changed = labels != mf
    grey = np.repeat(color.rgb2gray(proj)[..., None], 3, -1) * 0.5 + 0.5
    grey[ndi.binary_dilation(changed, iterations=2)] = [0.85, 0.1, 0.1]
    _save_arr(grey, "pj_islands.jpg", 512)
    order = [k for k in np.argsort(pl[:, 0]) if k != len(pl) - 1 and (labels == k).mean() > 2e-3]
    for i, k in enumerate(order[:8]):
        sh = np.ones_like(x); sh[labels == k] = rgbpal[k]
        _save_arr(sh, f"pj_sheet{i}.jpg", 256)
    print("sheets", len(order), st)




def _grid(rows, name, cell=1.45, gap=0.03):
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr))
    for i, r in enumerate(rows):
        for j, x in enumerate(r):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if x is None:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#999", fontsize=8, transform=ax.transAxes)
                continue
            ax.imshow(x if not isinstance(x, (str, Path)) else Image.open(x).convert("RGB").resize((384, 384)))
    plt.subplots_adjust(wspace=gap, hspace=gap, left=0, right=1, top=1, bottom=0)
    fig.savefig(FIG / name, pil_kwargs={"quality": 90}, dpi=220)
    plt.close(fig)


def fig8():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    ref = {"ind_" + t["id"]: ROOT / t["reference"] for t in split["test_indomain"]}
    ids = ["ind_动物-18", "ind_人物-5", "ind_植物-29", "ind_日常-3", "repo_fish", "ood_rooster", "ood_tiger", "ood_teapot"]
    rows = []
    for cid in ids:
        rows.append([ROOT / f"outputs/contents/{cid}.png",
                     im(ref[cid], 384) if cid in ref else None,
                     gen("blora", cid, 0, 0), gen("blora_guide", cid, 0, 0), gen("prompt_cn", cid), gen("coll", cid),
                     gen("cutline", cid), gen("cutcraft", cid), projected(gen("cutcraft", cid), cid)])
    _grid(rows, "fig8_qualitative.jpg", cell=1.25)


def fig9():
    from scipy import ndimage as ndi
    from skimage import morphology
    ids = ["ood_carp_fish", "ind_人物-16", "ood_rooster", "ood_crane_bird", "ind_动物-鸟-3"]  # 2 best F1 + rooster, 2 lowest IoU
    rows = []
    for cid in ids:
        la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
        lines = la.mean(-1) < 0.85
        gt = morphology.skeletonize(lines)
        out = priors.load_rgb(gen("cutcraft", cid), 512)
        proj = np.asarray(projected(gen("cutcraft", cid), cid)).astype(np.float32) / 255
        pe = priors.edge_map(proj)
        # palette-distance heat map on the raw output
        _, de = priors.nearest_palette(priors.to_lab(out), PAL.lab_with_white)
        heat = _cmap(de, "jet", 0, 25)
        # overlay: drawing lines in dark green over the projected output
        ov = proj.copy()
        ov[ndi.binary_dilation(gt, iterations=1)] = [0.1, 0.75, 0.1]
        # error map on the drawing's lines: TP green, FN blue; FP (edges far from any line) red
        dt_gt = ndi.distance_transform_edt(~gt)
        dt_pe = ndi.distance_transform_edt(~pe)
        base = np.repeat((0.35 + 0.65 * np.asarray(Image.fromarray((proj * 255).astype(np.uint8)).convert("L"),
                                                   np.float32)[..., None] / 255), 3, -1)
        err = base.copy()
        fp = pe & (dt_gt > 3)
        tp = gt & (dt_pe <= 3)
        fn = gt & (dt_pe > 3)
        err[ndi.binary_dilation(fp, iterations=1)] = [0.9, 0.1, 0.1]
        err[ndi.binary_dilation(tp, iterations=1)] = [0.1, 0.8, 0.1]
        err[ndi.binary_dilation(fn, iterations=1)] = [0.1, 0.2, 0.95]
        rows.append([Image.fromarray((la * 255).astype(np.uint8)), Image.fromarray((out * 255).astype(np.uint8)),
                     Image.fromarray((heat * 255).astype(np.uint8)), Image.fromarray((ov * 255).astype(np.uint8)),
                     Image.fromarray((np.clip(err, 0, 1) * 255).astype(np.uint8))])
        r = priors.edge_f1(pe, lines, 3)
        print(cid, {k: round(v, 3) for k, v in r.items()}, "IoU", round(priors.silhouette_iou(proj, priors.silhouette_from_lineart(lines)), 3))
    _grid(rows, "fig9_errors.jpg", cell=1.9)




def traj_curve():
    d = json.loads((ROOT / "outputs/trajectory.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.3))
    for ax, key, lab in zip(axes, ["E", "ground", "pal", "flat"],
                            ["total $E$", "$E_{ground}$", "$E_{pal}$", "$E_{flat}$"]):
        for run, c in (("unguided", "#6b6b6b"), ("guided", "#F08A24")):
            s = [r["step"] for r in d[run]]
            ax.plot(s, [r[key] for r in d[run]], color=c, lw=1.6, label=run)
        ax.axvspan(0.15 * 50, 0.75 * 50, color="#FDEBD6", alpha=0.7, lw=0)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    plt.tight_layout(w_pad=0.6)
    fig.savefig(ASSET / "traj_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
