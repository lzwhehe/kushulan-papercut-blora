"""Visual-element table of the digital corpus (data-derived counterpart of Zou et al. 2025, Table 1).

Per catalogue category: number of images, left-right symmetry (median and share of works >= 0.9, from
corpus_stats.csv), foreground share, mean chroma, palette entropy and the three paper colours with the largest
share (swatches). No symbolic connotations are listed: the catalogue does not document them, and they are not
inferred here. Writes paper/sections/table_elements.tex (with \\definecolor lines) and prints the motif classes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATS = [("person", "Figures"), ("animals", "Animals"), ("plants", "Plants"), ("daily", "Daily objects"),
        ("window decoration", "Window flowers"), ("frame", "Borders"), ("Pattern symbols", "Pattern symbols")]


def cat_of(path):
    return "Pattern symbols" if "Pattern_symbols" in path else path.split("/")[-2]


def main():
    pal = json.loads((ROOT / "outputs/data/palette.json").read_text())
    hexes = pal["hex"]
    stats = pd.read_csv(ROOT / "outputs/data/corpus_stats.csv")
    stats["cat"] = stats.path.map(cat_of)
    H = np.load(ROOT / "outputs/data/corpus_palette_shares.npy")[:, :len(hexes)]
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    paths = [it["path"] for it in split["train"]] + split["held_out"]
    hcat = np.array([cat_of(p) for p in paths])
    lines = [r"\definecolor{pal%d}{HTML}{%s}" % (k, h.lstrip("#").upper()) for k, h in enumerate(hexes)]
    body = []
    for key, lab in CATS:
        s = stats[stats.cat == key]
        sh = H[hcat == key].mean(0); sh = sh / sh.sum()
        ent = float(-(sh * np.log2(sh + 1e-12)).sum())
        top = np.argsort(-sh)[:3]
        sw = " ".join(r"\textcolor{pal%d}{\rule{2.2mm}{2.2mm}}\,%d\%%" % (k, round(sh[k] * 100)) for k in top)
        body.append(f"{lab} & {len(s)} & {s.symmetry.median():.2f} ({(s.symmetry >= 0.9).mean() * 100:.0f}\\%) & "
                    f"{s.fg_frac.mean():.2f} & {s.chroma.mean():.0f} & {ent:.1f} & {sw} \\\\")
    (ROOT / "paper/sections/table_elements_colors.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "paper/sections/table_elements.tex").write_text("\n".join(body) + "\n")
    print("\n".join(body))
    ps = stats[stats.cat == "Pattern symbols"].path.str.split("/").str[-2].value_counts()
    print(len(ps), ", ".join(ps.index))


if __name__ == "__main__":
    main()
