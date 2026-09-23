"""Render the paper-style framework figure (SVG, and PDF/PNG via headless Chrome).

Only thumbnails of images already in the repository are embedded. Nothing in
the figure is a new model output, and no unverified score is drawn.

    python tools/figures/make_framework_figure.py            # SVG + PDF + PNG
    python tools/figures/make_framework_figure.py --svg-only

Thumbnails are produced with ImageMagick (`magick`) when available; otherwise
the source files are embedded unchanged (larger SVG).
"""
import argparse
import base64
import html
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "framework"
DS = ROOT / "KUSHULAN_dataset"
SYM = ROOT / "data/整理的数据-常用_副本/库淑兰-data-处理后/Pattern_symbols"
RES = ROOT / "docs/assets/results"

IMAGES = {
    "figure": (DS / "人物.jpg/2.jpg.jpg", 240),
    "animal": (DS / "动物.jpg/18.jpg.jpg", 240),
    "object": (DS / "日常.jpg/3.jpg.jpg", 240),
    "plant": (DS / "植物.jpg/2.jpg.jpg", 240),
    "window": (DS / "窗花.jpg/1.jpg.jpg", 240),
    "border": (DS / "边框.jpg/3.jpg.jpg", 240),
    "work1": (DS / "人物.jpg/8.jpg.jpg", 240),
    "work2": (DS / "人物.jpg/5.jpg.jpg", 240),
    "fishcolor": (DS / "动物.jpg/34.jpg.jpg", 240),
    "fishline": (RES / "fish-content.png", 320),
    "fishstyle": (RES / "fish-style.png", 320),
    "fishout": (RES / "fish-result.jpeg", 320),
    "birdline": (RES / "bird-content.png", 240),
    "birdstyle": (RES / "bird-style.jpeg", 240),
    "birdout": (RES / "bird-result.jpeg", 240),
    "motifline": (RES / "motif-content.png", 240),
    "motifstyle": (RES / "motif-style.jpeg", 240),
    "motifout": (RES / "motif-result.jpeg", 240),
}
SYMBOLS = ["Bird", "Butterfly", "HangingCoin", "Plum", "Sun", "Tree", "Water"]
for _s in SYMBOLS:
    _d = SYM / _s
    if _d.is_dir():
        IMAGES["sym_" + _s] = (sorted(p for p in _d.iterdir() if p.is_file())[0], 160)

# Palette: muted, print- and colour-blind-friendly.
INK = "#1F2430"
MUTED = "#5B6472"
RULE = "#C9CED6"
PANEL = "#F7F8FA"
CONTENT = "#2F6DB5"      # content block / structure
CONTENT_BG = "#E6EEF8"
STYLE = "#C4492E"        # style block / appearance
STYLE_BG = "#F9E7E2"
FROZEN = "#8E99A8"
FROZEN_BG = "#E9ECF0"
DATA = "#2E8B7A"
DATA_BG = "#E3F2EF"
EVAL = "#8A6D1F"
EVAL_BG = "#F6F0DF"
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
SERIF = "Times New Roman, Times, serif"


def load_images():
    data = {}
    magick = shutil.which("magick")
    with tempfile.TemporaryDirectory() as tmp:
        for key, (path, size) in IMAGES.items():
            if not path.is_file() or path.stat().st_size < 1024:
                raise FileNotFoundError(f"{path} missing or still an LFS pointer; run git lfs pull first")
            if magick:
                out = Path(tmp) / f"{key}.jpg"
                trim = [] if key.endswith(("out", "style")) else ["-trim", "+repage"]
                subprocess.run([magick, str(path), "-background", "white", "-alpha", "remove", "-alpha", "off",
                                *trim, "-resize", f"{size}x{size}", "-gravity", "center",
                                "-extent", f"{size}x{size}", "-quality", "86", str(out)], check=True)
                raw, mime = out.read_bytes(), "image/jpeg"
            else:
                raw = path.read_bytes()
                mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
            data[key] = f"data:{mime};base64," + base64.b64encode(raw).decode()
    return data


class SVG:
    def __init__(self, w, h, images):
        self.w, self.h, self.img, self.parts = w, h, images, []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.2, r=6, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" opacity="{opacity}"{d}/>')

    def text(self, x, y, s, size=13, fill=INK, weight="normal", anchor="start", family=SANS, style="normal",
             raw=False):
        body = s if raw else html.escape(s)
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
                 f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">{body}</text>')

    def math(self, x, y, s, size=15, anchor="start", fill=INK):
        self.text(x, y, s, size=size, family=SERIF, anchor=anchor, fill=fill, raw=True)

    def image(self, key, x, y, w, h=None, border=RULE, r=4):
        h = h or w
        cid = f"clip{len(self.parts)}"
        self.add(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath>')
        self.add(f'<image href="{self.img[key]}" x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>')
        if border:
            self.rect(x, y, w, h, stroke=border, sw=1, r=r)

    def line(self, x1, y1, x2, y2, stroke=MUTED, sw=1.6, arrow=True, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{stroke[1:]})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, stroke=MUTED, sw=1.6, arrow=True, dash=None, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{stroke[1:]})"' if arrow else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{da}{m}/>')

    def snow(self, cx, cy, s=6, color=FROZEN):
        for a, b in ((0, 1), (0.866, 0.5), (0.866, -0.5)):
            self.add(f'<line x1="{cx - a * s}" y1="{cy - b * s}" x2="{cx + a * s}" y2="{cy + b * s}" '
                     f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>')

    def flame(self, cx, cy, s=7, color="#E0662B"):
        self.add(f'<path d="M{cx} {cy - s} C{cx + s * .9} {cy - s * .2} {cx + s * .7} {cy + s * .9} {cx} {cy + s} '
                 f'C{cx - s * .8} {cy + s * .9} {cx - s * .8} {cy} {cx - s * .2} {cy - s * .35} '
                 f'C{cx - s * .1} {cy + s * .1} {cx + s * .1} {cy + s * .2} {cx + s * .15} {cy} '
                 f'C{cx + s * .25} {cy - s * .3} {cx + s * .05} {cy - s * .6} {cx} {cy - s}Z" fill="{color}"/>')

    def panel(self, x, y, w, h, tag, title, accent, bg):
        self.rect(x, y, w, h, fill=PANEL, stroke=RULE, sw=1.2, r=10)
        self.rect(x, y, w, 34, fill=bg, r=10)
        self.rect(x, y + 20, w, 14, fill=bg, r=0)
        self.add(f'<line x1="{x}" y1="{y + 34}" x2="{x + w}" y2="{y + 34}" stroke="{RULE}" stroke-width="1"/>')
        self.text(x + 14, y + 23, f"({tag})", 16, accent, "bold")
        self.text(x + 46, y + 23, title, 16, INK, "bold")

    def chip(self, x, y, w, h, label, fg, bg, size=12, weight="bold", dash=None):
        self.rect(x, y, w, h, fill=bg, stroke=fg, sw=1.2, r=h / 2 if h < 30 else 6, dash=dash)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size, fg, weight, "middle")

    def render(self):
        markers = "".join(
            f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>'
            for c in (MUTED, CONTENT, STYLE, DATA, EVAL, INK, FROZEN))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}"><defs>{markers}</defs><rect width="100%" height="100%" fill="white"/>'
                + "".join(self.parts) + "</svg>")


def unet(s, x, y, hl=True):
    """SDXL UNet silhouette; returns centre of the highlighted up_blocks.0."""
    heights = [120, 96, 72, 56, 72, 96, 120]
    names = ["", "", "", "mid", "", "", ""]
    bw, gap = 20, 9
    cy = y + 62
    centres = []
    for i, bh in enumerate(heights):
        bx = x + i * (bw + gap)
        is_up0 = hl and i == 4
        fill, stroke = (("#FFF6E0", "#D39A1E") if is_up0 else (FROZEN_BG, FROZEN))
        s.rect(bx, cy - bh / 2, bw, bh, fill=fill, stroke=stroke, sw=1.8 if is_up0 else 1.1, r=4)
        if names[i]:
            s.text(bx + bw / 2, cy + bh / 2 + 14, names[i], 10, MUTED, anchor="middle")
        centres.append((bx + bw / 2, cy))
    for i in range(3):
        a, b = centres[i], centres[6 - i]
        top = cy - heights[i] / 2 - 10 - i * 0
        s.path(f"M{a[0]} {cy - heights[i] / 2} L{a[0]} {top - 8 + i * 12} L{b[0]} {top - 8 + i * 12} "
               f"L{b[0]} {cy - heights[i] / 2}", stroke=RULE, sw=1, arrow=False, dash="3 3")
    s.snow(x + 3 * (bw + gap) + bw / 2, cy - 44)
    s.text(x + 4 * (bw + gap) + bw / 2, cy + heights[4] / 2 + 14, "up₀", 10.5, "#B07D10", "bold", "middle")
    return centres[4], bw, heights[4]


def build(images):
    s = SVG(2000, 900, images)

    # ------------------------------------------------------------------ (a)
    ax, ay, aw, ah = 16, 16, 560, 868
    s.panel(ax, ay, aw, ah, "a", "Semi-structured Data Curation", DATA, DATA_BG)

    # curation chain: works -> element -> line art
    y0 = ay + 58
    for k, off in (("work2", 10), ("work1", 5), ("figure", 0)):
        s.image(k, ax + 24 + off, y0 + off, 118)
    s.text(ax + 88, y0 + 150, "Ku Shulan papercuts", 12, MUTED, anchor="middle")
    s.line(ax + 158, y0 + 64, ax + 206, y0 + 64, DATA)
    s.text(ax + 182, y0 + 50, "segment", 10.5, MUTED, anchor="middle")
    s.image("fishcolor", ax + 212, y0 + 5, 118)
    s.text(ax + 271, y0 + 150, "element", 12, MUTED, anchor="middle")
    s.math(ax + 271, y0 + 168, "<tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>col</tspan>", 14, "middle")
    s.line(ax + 336, y0 + 64, ax + 404, y0 + 64, DATA)
    s.text(ax + 370, y0 + 44, "line-art", 10.5, MUTED, anchor="middle")
    s.text(ax + 370, y0 + 57, "+ thicken", 10.5, MUTED, anchor="middle")
    s.image("fishline", ax + 410, y0 + 5, 118)
    s.text(ax + 469, y0 + 150, "line art", 12, MUTED, anchor="middle")
    s.math(ax + 469, y0 + 168, "<tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>line</tspan>", 14, "middle")

    # caption with content / style tokens
    cy0 = y0 + 186
    s.rect(ax + 20, cy0, aw - 40, 132, fill="white", stroke=RULE, r=6)
    s.text(ax + 32, cy0 + 20, "Structured caption", 12.5, INK, "bold")
    s.math(ax + 150, cy0 + 20, "<tspan font-style='italic'>y</tspan> = (<tspan fill='%s' font-style='italic'>y</tspan><tspan baseline-shift='sub' font-size='10' fill='%s'>c</tspan>, <tspan fill='%s' font-style='italic'>y</tspan><tspan baseline-shift='sub' font-size='10' fill='%s'>s</tspan>)" % (CONTENT, CONTENT, STYLE, STYLE), 14)
    rows = [
        [("“The kushulan animal papercut is composed of geometric figures.", MUTED)],
        [("animal, fish, ", CONTENT), ("round_shapes, arc_shapes, ", CONTENT), ("simple_design, ", CONTENT)],
        [("flat_design, red_background, green_accents, ", STYLE), ("yellow_accents,", STYLE)],
        [("red_yellow_combinations, geometric_patterns, sharp_lineart”", STYLE)],
    ]
    for i, row in enumerate(rows):
        spans = "".join(f'<tspan fill="{c}">{html.escape(t)}</tspan>' for t, c in row)
        s.add(f'<text x="{ax + 32}" y="{cy0 + 44 + i * 18}" font-family="Consolas, Menlo, monospace" '
              f'font-size="11.2">{spans}</text>')
    s.chip(ax + 32, cy0 + 108, 110, 17, "content tokens", CONTENT, CONTENT_BG, 10)
    s.chip(ax + 150, cy0 + 108, 132, 17, "style / symbol tokens", STYLE, STYLE_BG, 10)
    s.text(ax + aw - 30, cy0 + 121, "source: DOCX prompt records", 10, MUTED, anchor="end", style="italic")

    # categories
    gy = cy0 + 150
    s.text(ax + 22, gy + 4, "Six element categories", 13, INK, "bold")
    s.text(ax + aw - 22, gy + 4, "179 images", 12, MUTED, anchor="end")
    cats = [("figure", "Figure", 30), ("animal", "Animal", 46), ("object", "Object", 25),
            ("plant", "Plant", 38), ("window", "Window flower", 20), ("border", "Border", 20)]
    tw = 80
    for i, (k, name, n) in enumerate(cats):
        x = ax + 22 + i * (tw + 9.5)
        s.image(k, x, gy + 14, tw)
        s.text(x + tw / 2, gy + tw + 30, name, 11, INK, anchor="middle")
        s.text(x + tw / 2, gy + tw + 44, f"n = {n}", 10.5, MUTED, anchor="middle")

    # symbols
    sy = gy + 162
    s.text(ax + 22, sy + 4, "Pattern-symbol vocabulary", 13, INK, "bold")
    s.text(ax + aw - 22, sy + 4, "21 classes · 103 images", 12, MUTED, anchor="end")
    syms = [k for k in SYMBOLS if "sym_" + k in images]
    sw_ = 64
    for i, k in enumerate(syms):
        x = ax + 22 + i * (sw_ + 11.5)
        s.image("sym_" + k, x, sy + 14, sw_)
        s.text(x + sw_ / 2, sy + sw_ + 29, {"HangingCoin": "Coin"}.get(k, k), 10.5, MUTED, anchor="middle")

    # record schema
    ry = sy + 122
    s.rect(ax + 20, ry, aw - 40, 108, fill=DATA_BG, stroke=DATA, sw=1.3, r=8)
    s.text(ax + 36, ry + 24, "Record", 13, DATA, "bold")
    s.math(ax + 96, ry + 25, "<tspan font-style='italic'>r</tspan><tspan baseline-shift='sub' font-size='10'>i</tspan> = ⟨ <tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>line</tspan>, <tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>col</tspan>, <tspan font-style='italic'>c</tspan>, <tspan font-style='italic'>y</tspan>, <tspan font-style='italic'>g</tspan> ⟩", 16)
    s.text(ax + 36, ry + 50, "c: category / symbol class    g: source-artwork group", 11.5, MUTED)
    s.text(ax + 36, ry + 70, "282 curated images (179 elements + 103 symbols);", 11.5, INK)
    s.text(ax + 36, ry + 88, "splits grouped by g to avoid near-duplicate leakage", 11.5, INK)

    # legend
    ly = ry + 136
    s.add(f'<line x1="{ax + 20}" y1="{ly - 18}" x2="{ax + aw - 20}" y2="{ly - 18}" stroke="{RULE}"/>')
    s.snow(ax + 32, ly, 6)
    s.text(ax + 44, ly + 4, "frozen", 11.5, MUTED)
    s.flame(ax + 108, ly, 7)
    s.text(ax + 120, ly + 4, "trainable (LoRA)", 11.5, MUTED)
    s.rect(ax + 232, ly - 7, 14, 14, fill=CONTENT_BG, stroke=CONTENT, r=3)
    s.text(ax + 252, ly + 4, "content / structure", 11.5, MUTED)
    s.rect(ax + 378, ly - 7, 14, 14, fill=STYLE_BG, stroke=STYLE, r=3)
    s.text(ax + 398, ly + 4, "style / appearance", 11.5, MUTED)

    # ------------------------------------------------------------------ (b)
    bx, by, bw_, bh = 592, 16, 820, 470
    s.panel(bx, by, bw_, bh, "b", "Block-wise B-LoRA Fine-tuning on SDXL", CONTENT, CONTENT_BG)

    # inputs
    def stack(keys, x, y, size, color):
        for i, k in enumerate(keys):
            s.image(k, x + i * 12, y + i * 10, size, border=color if i == len(keys) - 1 else RULE)

    iy1, iy2 = by + 60, by + 262
    s.text(bx + 20, iy1 + 2, "Content run", 13, CONTENT, "bold")
    stack(["motifline", "birdline", "fishline"], bx + 20, iy1 + 12, 96, CONTENT)
    s.chip(bx + 20, iy1 + 140, 172, 26, "“A [c] papercut fish”", CONTENT, "white", 11, "normal", dash="4 3")
    s.math(bx + 158, iy1 + 70, "{<tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>line</tspan>, <tspan font-style='italic'>y</tspan><tspan baseline-shift='sub' font-size='10'>c</tspan>}", 15)

    s.text(bx + 20, iy2 + 2, "Style run", 13, STYLE, "bold")
    stack(["motifstyle", "birdstyle", "fishstyle"], bx + 20, iy2 + 12, 96, STYLE)
    s.chip(bx + 20, iy2 + 140, 172, 26, "enriched style caption", STYLE, "white", 11, "normal", dash="4 3")
    s.math(bx + 158, iy2 + 70, "{<tspan font-style='italic'>x</tspan><tspan baseline-shift='super' font-size='10'>col</tspan>, <tspan font-style='italic'>y</tspan><tspan baseline-shift='sub' font-size='10'>s</tspan>}", 15)

    # UNet
    ux, uy = bx + 262, by + 150
    s.text(ux + 100, uy - 44, "SDXL UNet (frozen)", 12.5, INK, "bold", "middle")
    (upx, upy), ubw, ubh = unet(s, ux, uy)
    s.line(bx + 225, iy1 + 70, ux - 8, uy + 50, CONTENT, 1.6)
    s.line(bx + 225, iy2 + 70, ux - 8, uy + 76, STYLE, 1.6)
    s.math(ux + 100, uy + 170, "ℒ = 𝔼<tspan baseline-shift='sub' font-size='10'>z,ε,t</tspan> ‖ ε − ε<tspan baseline-shift='sub' font-size='10'>θ+Δθ</tspan>(<tspan font-style='italic'>z</tspan><tspan baseline-shift='sub' font-size='10'>t</tspan>, <tspan font-style='italic'>t</tspan>, τ(<tspan font-style='italic'>y</tspan>)) ‖²", 15, "middle")
    s.text(ux + 100, uy + 190, "denoising objective; only LoRA factors are updated", 10.5, MUTED, anchor="middle")

    # zoom callout of up_blocks.0
    zx, zy, zw, zh = bx + 520, by + 52, 284, 300
    s.path(f"M{upx + ubw / 2} {upy - ubh / 2} L{zx} {zy + 20}", stroke="#D39A1E", sw=1, arrow=False, dash="3 3")
    s.path(f"M{upx + ubw / 2} {upy + ubh / 2} L{zx} {zy + zh - 10}", stroke="#D39A1E", sw=1, arrow=False, dash="3 3")
    s.rect(zx, zy, zw, zh, fill="#FFFCF3", stroke="#D39A1E", sw=1.3, r=8)
    s.add(f'<text x="{zx + 12}" y="{zy + 20}" font-family="Consolas, Menlo, monospace" font-size="11.5" '
          f'fill="{INK}">unet.up_blocks.0</text>')
    blocks = [("attentions.0", "content  W", "c", CONTENT, CONTENT_BG, True),
              ("attentions.1", "style  W", "s", STYLE, STYLE_BG, True),
              ("attentions.2", "frozen", "", FROZEN, FROZEN_BG, False)]
    for i, (nm, role, sub, fg, bg, lora) in enumerate(blocks):
        yy = zy + 34 + i * 62
        s.rect(zx + 12, yy, zw - 24, 52, fill=bg, stroke=fg, sw=1.3, r=6)
        s.add(f'<text x="{zx + 22}" y="{yy + 20}" font-family="Consolas, Menlo, monospace" font-size="11" '
              f'fill="{fg}">{nm}</text>')
        s.text(zx + 22, yy + 40, "Transformer ×10", 10.5, MUTED)
        if lora:
            s.math(zx + zw - 22, yy + 22, f"<tspan fill='{fg}' font-weight='bold'>{role.split()[0]}</tspan>", 13, "end")
            s.math(zx + zw - 22, yy + 42, f"<tspan font-style='italic'>W</tspan><tspan baseline-shift='sub' font-size='9'>0</tspan> + <tspan font-style='italic' fill='{fg}'>B</tspan><tspan baseline-shift='sub' font-size='9' fill='{fg}'>{sub}</tspan><tspan font-style='italic' fill='{fg}'>A</tspan><tspan baseline-shift='sub' font-size='9' fill='{fg}'>{sub}</tspan>", 14, "end")
            s.flame(zx + zw - 108, yy + 17, 6.5)
        else:
            s.snow(zx + zw - 30, yy + 26)
    s.text(zx + 12, zy + 236, "LoRA on to_q, to_k, to_v, to_out", 11, INK)
    s.math(zx + 12, zy + 256, "rank <tspan font-style='italic'>r</tspan> = 64,  <tspan font-style='italic'>A</tspan> ∈ ℝ<tspan baseline-shift='super' font-size='9'>r×d</tspan>, <tspan font-style='italic'>B</tspan> ∈ ℝ<tspan baseline-shift='super' font-size='9'>d×r</tspan>", 13)
    s.text(zx + 12, zy + 278, "all other UNet / text-encoder weights frozen", 10.5, MUTED)

    # exported adapters
    ey = by + 380
    for i, (name, tag, col) in enumerate((("ksl_content.safetensors", "run 2", CONTENT),
                                           ("ksl_style.safetensors", "run 1", STYLE))):
        x = bx + 262 + i * 276
        s.rect(x, ey, 262, 70, fill="white", stroke=col, sw=1.3, r=6)
        s.add(f'<text x="{x + 12}" y="{ey + 20}" font-family="Consolas, Menlo, monospace" font-size="11.5" '
              f'fill="{col}" font-weight="bold">{name}</text>')
        s.text(x + 250, ey + 20, tag, 10.5, MUTED, anchor="end")
        s.rect(x + 12, ey + 32, 116, 26, fill=CONTENT_BG, stroke=CONTENT, sw=1, r=4)
        s.text(x + 70, ey + 49, "attn.0 · 160 T", 10.5, CONTENT, anchor="middle")
        s.rect(x + 134, ey + 32, 116, 26, fill=STYLE_BG, stroke=STYLE, sw=1, r=4)
        s.text(x + 192, ey + 49, "attn.1 · 160 T", 10.5, STYLE, anchor="middle")
    s.line(ux + 100, uy + 198, ux + 100, ey - 4, MUTED, 1.4)
    for x_ in (bx + 393, bx + 669):
        s.line(x_, ey + 72, x_, 500, MUTED, 1.4)
    s.text(ux + 106, ey - 12, "export Δθ", 10.5, MUTED)

    # ------------------------------------------------------------------ (c)
    cx, cy_, cw, ch = 592, 502, 820, 382
    s.panel(cx, cy_, cw, ch, "c", "Style–Content Composition at Inference", STYLE, STYLE_BG)

    top = cy_ + 60
    # adapters -> filters
    for i, (lbl, keep, col, bg, alpha, sub) in enumerate((
            ("Δθ from run 2", "keep attn.0", CONTENT, CONTENT_BG, "α", "c"),
            ("Δθ from run 1", "keep attn.1", STYLE, STYLE_BG, "α", "s"))):
        yy = top + i * 118
        s.rect(cx + 20, yy, 150, 84, fill="white", stroke=col, sw=1.3, r=6)
        s.text(cx + 95, yy + 22, lbl, 12, col, "bold", "middle")
        s.rect(cx + 34, yy + 34, 58, 36, fill=CONTENT_BG, stroke=CONTENT, sw=1,
               r=4, opacity=1 if i == 0 else 0.3)
        s.rect(cx + 98, yy + 34, 58, 36, fill=STYLE_BG, stroke=STYLE, sw=1, r=4, opacity=1 if i == 1 else 0.3)
        s.text(cx + 63, yy + 57, "attn.0", 10.5, CONTENT, anchor="middle")
        s.text(cx + 127, yy + 57, "attn.1", 10.5, STYLE, anchor="middle")
        s.line(cx + 172, yy + 42, cx + 222, yy + 42, col)
        s.rect(cx + 226, yy + 22, 108, 40, fill=bg, stroke=col, sw=1.2, r=6)
        s.text(cx + 280, yy + 38, keep, 11, col, "bold", "middle")
        s.math(cx + 280, yy + 55, f"× <tspan font-style='italic'>{alpha}</tspan><tspan baseline-shift='sub' font-size='9'>{sub}</tspan>", 13, "middle")
    # merge
    mx, my = cx + 380, top + 101
    s.add(f'<circle cx="{mx}" cy="{my}" r="15" fill="white" stroke="{INK}" stroke-width="1.5"/>')
    s.text(mx, my + 6, "+", 20, INK, "bold", "middle")
    s.path(f"M{cx + 334} {top + 42} C{mx - 10} {top + 42} {mx} {top + 60} {mx} {my - 17}", CONTENT)
    s.path(f"M{cx + 334} {top + 160} C{mx - 10} {top + 160} {mx} {top + 142} {mx} {my + 17}", STYLE)
    s.math(mx + 34, my + 76, "Δθ* = <tspan font-style='italic'>α</tspan><tspan baseline-shift='sub' font-size='9'>c</tspan>Δθ<tspan baseline-shift='sub' font-size='9'>c</tspan> ⊕ <tspan font-style='italic'>α</tspan><tspan baseline-shift='sub' font-size='9'>s</tspan>Δθ<tspan baseline-shift='sub' font-size='9'>s</tspan>", 14, "middle")

    # SDXL + prompt + output
    gx = cx + 430
    s.line(mx + 17, my, gx + 8, my, INK)
    s.rect(gx + 10, my - 52, 128, 104, fill=FROZEN_BG, stroke=FROZEN, sw=1.2, r=8)
    s.text(gx + 74, my - 22, "SDXL", 14, INK, "bold", "middle")
    s.snow(gx + 118, my - 36)
    s.text(gx + 74, my - 2, "+ Δθ*", 12.5, INK, anchor="middle")
    s.text(gx + 74, my + 20, "text-to-image", 10.5, MUTED, anchor="middle")
    s.text(gx + 74, my + 34, "sampling", 10.5, MUTED, anchor="middle")
    s.chip(gx - 20, my - 100, 188, 26, "“A [v] in ksl style”", INK, "white", 11, "normal", dash="4 3")
    s.line(gx + 74, my - 72, gx + 74, my - 56, INK, 1.3)
    s.line(gx + 140, my, gx + 180, my, INK)
    s.image("fishout", gx + 184, my - 80, 160, border=INK)
    s.text(gx + 264, my + 98, "generated", 12, INK, anchor="middle")
    s.math(gx + 264, my + 116, "<tspan font-style='italic'>x̂</tspan>", 15, "middle")

    # provenance strip
    py_ = cy_ + ch - 60
    s.image("fishline", cx + 20, py_ - 4, 52, border=CONTENT)
    s.image("fishstyle", cx + 80, py_ - 4, 52, border=STYLE)
    s.text(cx + 144, py_ + 16, "structure is inherited from the content adapter, colour / ornament from the", 11, MUTED)
    s.text(cx + 144, py_ + 33, "style adapter; the two blocks are swapped independently across runs.", 11, MUTED)

    # ------------------------------------------------------------------ (d)
    dx, dy, dw, dh = 1428, 16, 556, 868
    s.panel(dx, dy, dw, dh, "d", "Evaluation Protocol", EVAL, EVAL_BG)
    s.line(1412, 694, 1426, 694, INK, 1.6)

    hy = dy + 58
    heads = [("content  x", "line", CONTENT), ("style  x", "col", STYLE), ("output  x̂", "", INK)]
    iw = 150
    for j, (h, sup, col) in enumerate(heads):
        xx = dx + 26 + j * (iw + 16) + iw / 2
        s.math(xx, hy, f"<tspan fill='{col}'>{h}</tspan><tspan baseline-shift='super' font-size='9' fill='{col}'>{sup}</tspan>", 14, "middle")
    for i, pre in enumerate(("fish", "bird", "motif")):
        for j, suf in enumerate(("line", "style", "out")):
            s.image(pre + suf, dx + 26 + j * (iw + 16), hy + 12 + i * (iw + 12), iw,
                    border=(CONTENT, STYLE, INK)[j])
    s.text(dx + dw / 2, hy + 12 + 3 * (iw + 12) + 8, "archived qualitative results (7.5 run); not cherry-picked for this figure",
           10.5, MUTED, anchor="middle", style="italic")

    my0 = hy + 12 + 3 * (iw + 12) + 26
    mets = [
        ("Structure", "DINO(x̂, x<sup>line</sup>) · Edge-F1 / Sil. IoU †", CONTENT, CONTENT_BG),
        ("Style", "CLIP-I(x̂, x<sup>col</sup>) · colour-palette dist.", STYLE, STYLE_BG),
        ("Text", "CLIP-T(x̂, y)", INK, FROZEN_BG),
        ("Human", "structure · element · style · symbol (1–5), Kendall’s W", EVAL, EVAL_BG),
    ]
    for i, (k, v, col, bg) in enumerate(mets):
        yy = my0 + i * 34
        s.rect(dx + 22, yy, dw - 44, 28, fill=bg, stroke=col, sw=1, r=5)
        s.text(dx + 34, yy + 19, k, 12, col, "bold")
        v = v.replace("<sup>", "<tspan baseline-shift='super' font-size='8'>").replace("</sup>", "</tspan>")
        s.text(dx + 112, yy + 19, v, 11.5, INK, raw=True)
    s.text(dx + 22, my0 + 150, "† pixel-aligned; reported with a mismatched-content control and", 10.5, MUTED)
    s.text(dx + 22, my0 + 165, "   counted only when matched > control (95% bootstrap CI)", 10.5, MUTED)
    s.text(dx + dw / 2, my0 + 188, "tools/generate.py (fixed seeds) → tools/evaluate.py (per-sample CSV)", 10.5, MUTED,
           anchor="middle", style="italic")

    return s.render()


def render_with_chrome(svg_path, pdf_path, png_path, w, h):
    chrome = next((p for p in (shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"),
                               r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                   if p and Path(p).exists()), None)
    if not chrome:
        print("Chrome/Edge not found; wrote SVG only")
        return
    page = svg_path.with_suffix(".html")
    page.write_text(f"<!doctype html><meta charset='utf-8'><style>@page{{size:{w}px {h}px;margin:0}}"
                    f"html,body{{margin:0;padding:0}}img{{display:block;width:{w}px;height:{h}px}}</style>"
                    f"<img src='{svg_path.name}'>", encoding="utf-8")
    url = page.resolve().as_uri()
    common = [chrome, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--hide-scrollbars"]
    subprocess.run(common + [f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer", url], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(common + [f"--screenshot={png_path}", f"--window-size={w},{h}",
                             "--force-device-scale-factor=2", url], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    page.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--svg-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    svg = build(load_images())
    svg_path = args.out / "framework.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path} ({len(svg) // 1024} KiB)")
    if not args.svg_only:
        render_with_chrome(svg_path, args.out / "framework.pdf", args.out / "framework.png", 2000, 900)
        print(f"wrote {args.out / 'framework.pdf'} and {args.out / 'framework.png'}")
