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
    _save_arr(_cmap(de, "viridis", 0, 20), "cp_de.jpg", 512)
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
        _save_arr(_cmap(dx, "viridis", 0, 20), f"cg_{tag}_pal.jpg", 512)
        gx = np.sqrt(sum(ndi.sobel(lx[..., c], 1) ** 2 + ndi.sobel(lx[..., c], 0) ** 2 for c in range(3))) / 8
        _save_arr(_cmap(1 - np.exp(-(gx ** 2) / 64), "magma", 0, 1), f"cg_{tag}_flat.jpg", 512)
        g = ground_of(cid)
        dwhite = np.linalg.norm(lx - np.array([100, 0, 0]), axis=-1) * g
        _save_arr(_cmap(dwhite, "viridis", 0, 60), f"cg_{tag}_ground.jpg", 512)
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




def _grid(rows, name, cell=1.45, gap=0.03, col_labels=None, row_labels=None, header_fs=7):
    nr, nc = len(rows), len(rows[0])
    top = 0.35 if col_labels else 0.0
    left = 0.55 if row_labels else 0.0
    fig = plt.figure(figsize=(cell * nc + left, cell * nr + top))
    gs = fig.add_gridspec(nr, nc, left=left / (cell * nc + left), right=1, bottom=0,
                          top=1 - top / (cell * nr + top), wspace=gap, hspace=gap)
    for i, r in enumerate(rows):
        for j, x in enumerate(r):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=header_fs, pad=2)
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], fontsize=header_fs - 0.5, rotation=90, labelpad=2)
            if x is None:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#999", fontsize=7, transform=ax.transAxes)
                continue
            ax.imshow(x if not isinstance(x, (str, Path)) else Image.open(x).convert("RGB").resize((384, 384)))
    fig.savefig(FIG / name, pil_kwargs={"quality": 90}, dpi=220)
    plt.close(fig)


def fig8():
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    ref = {"ind_" + t["id"]: ROOT / t["reference"] for t in split["test_indomain"]}
    cats = {c["id"]: c for c in json.loads((ROOT / "outputs/contents/contents.json").read_text())}
    # in-domain showcase: per category the drawing with the highest mean CLIP style of CutCraft-SDXL and CutCraft-Qwen
    # (seed 0), plus the two best remaining; all have an original. New subjects: Supplementary figure (figS_newsubjects).
    ids = ["ind_人物-3", "ind_动物-38", "ind_窗花-3", "ind_日常-25", "ind_植物-17", "ind_边框-12", "ind_动物-18", "ind_窗花-4"]
    cols = [("drawing", None), ("original", None), ("B-LoRA", "blora"), ("StyleAligned", "stylealigned"),
            ("InstantStyle", "instantstyle"), ("Qwen-Image-Edit\nzero-shot", "qwen_q0"),
            ("CutCraft", "cutcraft"), ("CutCraft\ncutting plan", "cutcraft+proj")]
    cols = [c for c in cols if c[1] is None or (G / c[1].replace("+proj", "")).exists()]
    rows, rlab = [], []
    for cid in ids:
        r = []
        for lab, m in cols:
            if lab == "drawing":
                r.append(ROOT / f"outputs/contents/{cid}.png")
            elif lab == "original":
                r.append(im(ref[cid], 384) if cid in ref else None)
            elif m.endswith("+proj"):
                r.append(projected(gen(m[:-5], cid), cid))
            elif m in ("blora", "instantstyle", "instantstyle_guide", "stylealigned"):
                r.append(gen(m, cid, 0, 0))
            else:
                r.append(gen(m, cid))
        rows.append(r)
        c = cats[cid]
        if c["set"] == "indomain":
            num = cid.rsplit("-", 1)[-1]
            rlab.append(f"{c['category']} {num}\n(in-domain)")
        else:
            rlab.append(c["subject"].replace("a ", "", 1) + ("\n(repository)" if c["set"] == "repo" else "\n(new subject)"))
    _grid(rows, "fig8_qualitative.jpg", cell=1.15, col_labels=[c[0] for c in cols], row_labels=rlab, header_fs=6.5)


def fig9():
    from matplotlib.lines import Line2D
    from scipy import ndimage as ndi
    from skimage import morphology
    ids = ["ood_carp_fish", "ind_动物-38", "ood_rooster", "ood_crane_bird", "ind_动物-鸟-3"]
    names = ["carp (highest plan F1)", "animal 38 (2nd plan F1)", "rooster", "crane (2nd lowest IoU)", "sparse bird (lowest IoU)"]
    TP, FN, FP = np.array([0, 114, 178]) / 255, np.array([230, 159, 0]) / 255, np.array([204, 121, 167]) / 255
    fig = plt.figure(figsize=(9.6, 10.6))
    gs = fig.add_gridspec(len(ids) + 1, 5, height_ratios=[1] * len(ids) + [0.12], hspace=0.05, wspace=0.03,
                          left=0.07, right=0.995, top=0.965, bottom=0.045)
    heads = ["line drawing", "CutCraft output", "palette distance (raw)", "cut projection + drawing", "error map on projection"]
    for i, cid in enumerate(ids):
        la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
        lines = la.mean(-1) < 0.85
        gt = morphology.skeletonize(lines)
        out = priors.load_rgb(gen("cutcraft", cid), 512)
        proj = np.asarray(projected(gen("cutcraft", cid), cid)).astype(np.float32) / 255
        pe = priors.edge_map(proj)
        _, de = priors.nearest_palette(priors.to_lab(out), PAL.lab_with_white)
        ov = proj.copy()
        ov[ndi.binary_dilation(gt, iterations=1)] = [0.0, 0.0, 0.0]
        dt_gt, dt_pe = ndi.distance_transform_edt(~gt), ndi.distance_transform_edt(~pe)
        base = np.repeat(0.55 + 0.45 * proj.mean(-1, keepdims=True), 3, -1)
        err = base.copy()
        fp = pe & (dt_gt > 3)
        yy, xx = np.indices(fp.shape)
        err[fp & (((yy + xx) // 2) % 2 == 0)] = FP  # dotted: extra (mostly decorative) cut edges
        err[ndi.binary_dilation(gt & (dt_pe <= 3), iterations=1)] = TP  # solid: reproduced drawing lines
        err[ndi.binary_dilation(gt & (dt_pe > 3), iterations=2)] = FN  # thick: missed drawing lines
        r = priors.edge_f1(pe, lines, 3)
        iou = priors.silhouette_iou(proj, priors.silhouette_from_lineart(lines))
        panels = [la, out, None, ov, err]
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            if j == 2:
                hm = ax.imshow(de, cmap="viridis", vmin=0, vmax=25)
            else:
                ax.imshow(np.clip(panels[j], 0, 1))
            if i == 0:
                ax.set_title(f"({'abcde'[j]}) {heads[j]}", fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(names[i] + f"\nrecall {r['edge_r']:.2f} · F1 {r['edge_f1']:.2f} · IoU {iou:.2f}", fontsize=7)
    cax = fig.add_subplot(gs[-1, 2])
    cb = fig.colorbar(hm, cax=cax, orientation="horizontal")
    cb.set_label("CIEDE2000 to nearest paper colour", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    lax = fig.add_subplot(gs[-1, 3:])
    lax.axis("off")
    lax.legend(handles=[Line2D([0], [0], color=TP, lw=2, label="drawing line reproduced (TP)"),
                        Line2D([0], [0], color=FN, lw=4, label="drawing line missed (FN)"),
                        Line2D([0], [0], color=FP, lw=2, ls=":", label="extra cut edge (FP, mostly decoration)")],
               loc="center", ncol=1, fontsize=7, frameon=False)
    fig.savefig(FIG / "fig9_errors.jpg", pil_kwargs={"quality": 90}, dpi=220)
    plt.close(fig)


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




def colourbars():
    """Stand-alone colour bars for the TikZ method figures (review #48)."""
    import matplotlib as mpl
    for name, cmap, vmin, vmax, label in [("cbar_de20", "viridis", 0, 20, r"CIEDE2000 to nearest paper colour"),
                                          ("cbar_ground60", "viridis", 0, 60, r"CIELAB distance to paper white"),
                                          ("cbar_flat", "magma", 0, 1, r"flatness penalty $1-e^{-|\nabla\ell|^2/\beta^2}$")]:
        fig, ax = plt.subplots(figsize=(2.2, 0.42))
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap), cax=ax,
                          orientation="horizontal")
        cb.set_label(label, fontsize=6.5, labelpad=1)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        fig.savefig(ASSET / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)



def figS7():
    """Conventional LoRA (all attention layers, same 257 images) with and without CraftGuide vs CutCraft."""
    ids = ["ood_rooster", "ind_人物-5", "ood_tiger", "ind_植物-29", "repo_fish", "ood_teapot"]
    cols = [("drawing", None), ("conv. LoRA\n+ ControlNet", "fulllora"), ("conv. LoRA\n+ CraftGuide", "fulllora_guide"),
            ("conv. LoRA + guide\n+ projection", "fulllora_guide+proj"), ("CutCraft", "cutcraft"), ("CutCraft\n+ projection", "cutcraft+proj")]
    rows = []
    for cid in ids:
        r = []
        for lab, m in cols:
            if m is None:
                r.append(ROOT / f"outputs/contents/{cid}.png")
            elif m.endswith("+proj"):
                r.append(projected(gen(m[:-5], cid), cid))
            else:
                r.append(gen(m, cid))
        rows.append(r)
    rlab = ["rooster", "person 5", "tiger", "plant 29", "fish (repository)", "teapot"]
    _grid(rows, "figS7_convlora.jpg", cell=1.3, col_labels=[c[0] for c in cols], row_labels=rlab, header_fs=6.5)


def figS_newsubjects():
    """All 12 new-subject drawings (no original exists): popular frameworks and the two instantiations, seed 0."""
    cats = {c["id"]: c for c in json.loads((ROOT / "outputs/contents/contents.json").read_text())}
    ids = [k for k in cats if cats[k]["set"] == "ood"]
    cols = [("drawing", None), ("B-LoRA", "blora"), ("StyleAligned", "stylealigned"), ("InstantStyle", "instantstyle"),
            ("Qwen-Image-Edit\nzero-shot", "qwen_q0"), ("CutCraft", "cutcraft"), ("CutCraft\ncutting plan", "cutcraft+proj")]
    rows = []
    for cid in ids:
        r = []
        for lab, m in cols:
            if m is None:
                r.append(ROOT / f"outputs/contents/{cid}.png")
            elif m.endswith("+proj"):
                r.append(projected(gen(m[:-5], cid), cid))
            elif m in ("blora", "instantstyle", "stylealigned"):
                r.append(gen(m, cid, 0, 0))
            else:
                r.append(gen(m, cid))
        rows.append(r)
    rlab = [cats[c]["subject"].replace("a ", "", 1) for c in ids]
    _grid(rows, "figS9_newsubjects.jpg", cell=1.0, col_labels=[c[0] for c in cols], row_labels=rlab, header_fs=6)

if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
