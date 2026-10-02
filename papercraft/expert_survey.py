"""Self-contained HTML questionnaire for the blinded expert study (Supplementary S5).

Uses the existing blinded pack: the 111 codes and key of results/cutcraft/study_materials/expert_pack/key_UNBLIND.csv
(20 drawings x 5 methods + 11 reference redrawings as anchors). Every stimulus is shown as a cutting plan beside
its line drawing; anchors are projected onto the paper stock in the same way as the generated designs, so that
they cannot be recognised by their finish. Six stimuli are repeated at the end (codes R01-R06) to estimate
within-rater consistency. The presentation order is shuffled per rater. Raters open the file in a browser
(no login, no network), progress is kept in the browser, and the answers are exported as a JSON/CSV file that
is sent back to the researchers.

Writes outputs/expert_survey/kushulan_expert_survey.html (contains reference redrawings: share only with raters)
and outputs/expert_survey/repeats.csv (which stimulus each repeat code shows; withheld from raters).
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
OUT = ROOT / "outputs/expert_survey"
TEMPLATE = Path(__file__).with_name("expert_survey_template.html")
N_REPEAT = 6
SIZE = 448


def b64jpg(arr_or_img):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray((np.clip(arr_or_img, 0, 1) * 255).astype(np.uint8))
    im = im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    pal = priors.Palette.load(ROOT / "outputs/data/palette.json")
    pal_rgb = np.clip(priors.color.lab2rgb(pal.lab_with_white[None])[0], 0, 1)
    split = json.loads((ROOT / "outputs/data/split.json").read_text())
    refs = {f"ind_{t['id']}": t["reference"] for t in split["test_indomain"]}
    key = pd.read_csv(PACK / "key_UNBLIND.csv")

    drawings, stimuli = {}, {}
    for r in key.itertuples():
        if r.drawing not in drawings:
            drawings[r.drawing] = b64jpg(Image.open(ROOT / f"outputs/contents/{r.drawing}.png"))
        if r.method == "original":
            ref = ImageOps.pad(Image.open(ROOT / refs[r.drawing]).convert("RGB"), (512, 512), color="white")
            proj, _, _, _ = priors.cut_project(np.asarray(ref).astype(np.float32) / 255, pal)
            img = proj
        else:
            img = plan_image(r.method, r.drawing, pal_rgb)
        stimuli[r.code] = {"drawing": f"D{list(drawings).index(r.drawing) + 1:02d}", "img": b64jpg(img)}
    dmap = {f"D{i + 1:02d}": v for i, v in enumerate(drawings.values())}

    rng = np.random.default_rng(2026)
    rep = sorted(rng.choice(key.code.to_numpy(), N_REPEAT, replace=False))
    repeats = [{"code": f"R{i + 1:02d}", "repeats": c} for i, c in enumerate(rep)]
    for r in repeats:
        stimuli[r["code"]] = dict(stimuli[r["repeats"]])

    data = {"version": "ksl-expert-v1", "stimuli": stimuli, "drawings": dmap,
            "main": key.code.tolist(), "repeat": [r["code"] for r in repeats]}
    html = TEMPLATE.read_text(encoding="utf8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kushulan_expert_survey.html").write_text(html, encoding="utf8")
    pd.DataFrame(repeats).to_csv(OUT / "repeats.csv", index=False)
    print(f"{len(key)} stimuli + {N_REPEAT} repeats, {len(dmap)} drawings;",
          f"{(OUT / 'kushulan_expert_survey.html').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
