"""Self-contained HTML questionnaires for the blinded expert study (Supplementary S5), short version (v2).

Design: the 20 drawings of the original pack (results/cutcraft/study_materials/expert_pack) are split into two
forms A and B of 10 drawings each (in-domain and transfer drawings stratified, catalogue categories spread).
In each form every drawing is shown as the cutting plan of four methods (ours, InstantStyle, the untrained
editing model, B-LoRA); five in-domain drawings per form add the reference redrawing as an anchor, projected
onto the paper stock like the generated designs; three stimuli are repeated at the end for within-rater
consistency. Three dimensions are rated on 1-5: style resemblance, faithfulness to the drawing, feasibility of
cutting and pasting. Each form has 48 items (about 15-20 minutes). The order is shuffled per rater.

Writes outputs/expert_survey/kushulan_expert_survey_{A,B}.html (contain reference redrawings: share only with
raters) and results/cutcraft/study_materials/expert_pack_v2/key_UNBLIND.csv (form, code, drawing, method,
repeat-of; withheld from raters).
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import priors
from expert_pack import plan_image

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results/cutcraft/study_materials/expert_pack"
KEY2 = ROOT / "results/cutcraft/study_materials/expert_pack_v2"
OUT = ROOT / "outputs/expert_survey"
TEMPLATE = Path(__file__).with_name("expert_survey_template.html")
METHODS = ["qwen_q1", "instantstyle", "qwen_q0", "blora"]
N_ANCHOR, N_REPEAT, SIZE = 5, 3, 448


def b64jpg(arr_or_img):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray((np.clip(arr_or_img, 0, 1) * 255).astype(np.uint8))
    im = im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def split_forms(drawings, rng):
    """Two forms of 10 drawings: in-domain drawings alternate by catalogue category, transfer drawings by shuffle."""
    ind = sorted(d for d in drawings if d.startswith("ind_"))
    tra = [d for d in drawings if not d.startswith("ind_")]
    ind = [ind[i] for i in rng.permutation(len(ind))]
    ind.sort(key=lambda d: d.split("_")[1].split("-")[0])  # group by category, random within
    tra = [tra[i] for i in rng.permutation(len(tra))]
    a, b = ind[0::2], ind[1::2]                                # 6 / 5 in-domain
    need_a = 10 - len(a)
    return a + tra[:need_a], b + tra[need_a:]


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    pal_rgb = np.clip(priors.color.lab2rgb(pal.lab_with_white[None])[0], 0, 1)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    refs = {f"ind_{t['id']}": t["reference"] for t in split["test_indomain"]}
    old = pd.read_csv(PACK / "key_UNBLIND.csv")
    drawings = old.drawing.drop_duplicates().tolist()
    rng = np.random.default_rng(2026)
    forms = dict(zip("AB", split_forms(drawings, rng)))

    OUT.mkdir(parents=True, exist_ok=True)
    KEY2.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for form, ds in forms.items():
        dmap, stimuli, items = {}, {}, []
        for i, d in enumerate(ds):
            dmap[f"D{i + 1:02d}"] = b64jpg(Image.open(ROOT / f"outputs/contents/{d}.png"))
            items += [(d, m) for m in METHODS]
        anchors = [d for d in ds if d.startswith("ind_")]
        anchors = [anchors[i] for i in sorted(rng.choice(len(anchors), min(N_ANCHOR, len(anchors)), replace=False))]
        items += [(d, "original") for d in anchors]
        order = rng.permutation(len(items))
        codes = []
        for n, i in enumerate(order, 1):
            d, m = items[i]
            code = f"{form}{n:02d}"
            if m == "original":
                ref = ImageOps.pad(Image.open(ROOT / refs[d]).convert("RGB"), (512, 512), color="white")
                img, _, _, _ = priors.cut_project(np.asarray(ref).astype(np.float32) / 255, pal)
            else:
                img = plan_image(m, d, pal_rgb)
            stimuli[code] = {"drawing": f"D{ds.index(d) + 1:02d}", "img": b64jpg(img)}
            key_rows.append({"form": form, "code": code, "drawing": d, "method": m, "repeat_of": ""})
            codes.append(code)
        reps = []
        for j, c in enumerate(sorted(rng.choice(codes, N_REPEAT, replace=False)), 1):
            rc = f"{form}R{j}"
            stimuli[rc] = dict(stimuli[c])
            src = next(r for r in key_rows if r["code"] == c)
            key_rows.append({**src, "code": rc, "repeat_of": c})
            reps.append(rc)
        data = {"version": f"ksl-expert-v2-{form}", "form": form, "stimuli": stimuli, "drawings": dmap,
                "main": codes, "repeat": reps}
        html = TEMPLATE.read_text(encoding="utf8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
        path = OUT / f"kushulan_expert_survey_{form}.html"
        path.write_text(html, encoding="utf8")
        print(f"form {form}: {len(ds)} drawings ({sum(d.startswith('ind_') for d in ds)} in-domain), "
              f"{len(codes)} items + {len(reps)} repeats; {path.stat().st_size / 1e6:.1f} MB")
    pd.DataFrame(key_rows).to_csv(KEY2 / "key_UNBLIND.csv", index=False)


if __name__ == "__main__":
    main()
