"""Review #35: CPU time of the delivered-plan export (enforced 1024-px projection + mm SVG), 10 CutCraft outputs."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import priors
from svg_validate import MIN_MM2, NECK_MM, PX_MM, RES, SIZE_MM, export_svg

ROOT = Path(__file__).resolve().parents[1]


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    files = sorted((ROOT / "outputs/gen/cutcraft").glob("*_seed0.png"))[:11]
    rows = []
    for n, f in enumerate(files):
        cid = f.name.split("_s_seed")[0]
        t0 = time.perf_counter()
        rgb = priors.load_rgb(f, RES)
        la = priors.load_rgb(ROOT / f"outputs/contents/{cid}.png", 512)
        g = priors.ground_from_lineart(la.mean(-1) < 0.85)
        g = np.asarray(Image.fromarray(g.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 0
        _, labels, _, _ = priors.cut_project(rgb, pal, min_area_frac=MIN_MM2 / SIZE_MM ** 2 * 1.1, ground=g,
                                             min_width_px=2 * max(1, int(round(NECK_MM / 2 / PX_MM))), enforce=True)
        t1 = time.perf_counter()
        export_svg(labels, pal, ROOT / "outputs/scratch/timing_export.svg", simplify=0.0)
        t2 = time.perf_counter()
        if n > 0:  # first run excluded (imports, caches)
            rows.append({"id": cid, "enforced_projection_s": t1 - t0, "svg_export_s": t2 - t1})
    out = {k: {"median": float(np.median([r[k] for r in rows])), "min": float(np.min([r[k] for r in rows])),
               "max": float(np.max([r[k] for r in rows]))} for k in rows[0] if k != "id"}
    (ROOT / "outputs/timing_export.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    (ROOT / "outputs/scratch").mkdir(parents=True, exist_ok=True)
    main()
