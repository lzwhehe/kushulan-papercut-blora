import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_records import parse_docx, split_terms, stem_number
from split import assign, group

try:
    import numpy as np
    from PIL import Image, ImageDraw
    import evaluate
except ImportError:  # image metrics need numpy + Pillow
    np = None

ROOT = Path(__file__).resolve().parents[1]


def make_docx(path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="w"><w:body>{body}</w:body></w:document>')


class RecordTests(unittest.TestCase):
    def test_parse_docx_tolerates_source_quirks(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.docx"
            make_docx(path, [
                '{&quot;file_name&quot;: &quot;1.jpg&quot;, &quot;text&quot;: &quot;A red fish.&quot;}',
                # opening quote of text missing (occurs in the source DOCX)
                '{&quot;file_name&quot;: &quot;plant_2.jpg&quot;, &quot;text&quot;: A blue bird.&quot;}',
                # closing brace missing (occurs in the source DOCX)
                '{&quot;file_name&quot;: &quot;3.jpg&quot;, &quot;text&quot;: &quot;A peach.',
                "unrelated paragraph",
            ])
            recs = parse_docx(path)
        self.assertEqual([r[0] for r in recs], ["1.jpg", "plant_2.jpg", "3.jpg"])
        self.assertEqual(recs[1][1], "A blue bird.")
        self.assertEqual([stem_number(r[0]) for r in recs], [1, 2, 3])
        self.assertEqual(stem_number("7..jpg.jpg"), 7)

    def test_split_terms_tag_caption(self):
        content, style = split_terms("The kushulan animal papercut is composed of geometric figures."
                                     "animal, fish, round_shapes, red_background, yellow_accents, flat_design")
        self.assertEqual(content, ["animal", "fish", "round_shapes"])
        self.assertEqual(style, ["red_background", "yellow_accents", "flat_design"])

    def test_committed_records_cover_all_images(self):
        recs = [json.loads(l) for l in (ROOT / "metadata/records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(recs), 282)
        self.assertEqual(sum(r["set"] == "element" for r in recs), 179)
        self.assertEqual(len({r["id"] for r in recs}), 282)


class SplitTests(unittest.TestCase):
    def test_duplicates_share_a_split(self):
        recs = [{"id": str(i), "sha256": "same" if i in (0, 7) else str(i), "category": "A" if i % 2 else "B"}
                for i in range(40)]
        groups = group(recs)
        self.assertEqual(groups[0], groups[7])
        splits = assign(recs, groups, seed=3)
        self.assertEqual(splits[0], splits[7])
        self.assertEqual(set(splits), {"train", "val", "test"})

    def test_near_duplicate_hash_and_aspect(self):
        recs = [{"id": str(i), "sha256": str(i), "category": "A"} for i in range(3)]
        hashes = [(0b1111, 1.0), (0b1110, 1.05), (0b1110, 3.0)]
        groups = group(recs, hashes, max_distance=1)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])  # same bits but different aspect ratio


@unittest.skipIf(np is None, "numpy/Pillow not installed")
class AgreementTests(unittest.TestCase):
    def test_kendalls_w(self):
        kendalls_w = evaluate.kendalls_w
        self.assertAlmostEqual(kendalls_w([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), 1.0)
        self.assertAlmostEqual(kendalls_w([[1, 2, 3], [3, 2, 1]]), 0.0)
        # textbook example with ties is bounded in [0, 1]
        w = kendalls_w([[1, 2, 2, 4], [2, 1, 3, 4], [1, 3, 3, 3]])
        self.assertTrue(0 <= w <= 1)


@unittest.skipIf(np is None, "numpy/Pillow not installed")
class MetricTests(unittest.TestCase):
    @staticmethod
    def drawing(shape, fill=None, size=128):
        im = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(im)
        (d.ellipse if shape == "circle" else d.rectangle)((24, 24, 104, 104), outline="black", width=3, fill=fill)
        return np.asarray(im, dtype=np.float32) / 255

    def test_aligned_beats_mismatched(self):
        line = self.drawing("circle")
        out = self.drawing("circle", fill=(200, 30, 30))
        other = self.drawing("square")
        self.assertGreater(evaluate.edge_f1(out, line)["edge_f1"], evaluate.edge_f1(out, other)["edge_f1"])
        self.assertGreater(evaluate.silhouette_iou(out, line), 0.9)
        self.assertLess(evaluate.silhouette_iou(out, other), evaluate.silhouette_iou(out, line))

    def test_palette_distance(self):
        red = self.drawing("circle", fill=(200, 30, 30))
        blue = self.drawing("circle", fill=(30, 30, 200))
        self.assertAlmostEqual(evaluate.palette_distance(red, red), 0.0, places=6)
        self.assertGreater(evaluate.palette_distance(red, blue), 0.5)

    def test_pairs_csv_with_control(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, shape, fill in (("c1", "circle", None), ("o1", "circle", (200, 30, 30)),
                                      ("c2", "square", None), ("o2", "square", (30, 30, 200))):
                Image.fromarray((self.drawing(shape, fill) * 255).astype("uint8")).save(d / f"{name}.png")
            with (d / "pairs.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "content", "style", "output"])
                w.writerow(["a", "c1.png", "o1.png", "o1.png"])
                w.writerow(["b", "c2.png", "o2.png", "o2.png"])
            rows = evaluate.evaluate_pairs(d / "pairs.csv", size=128)
            summary = evaluate.summarize(rows)
        self.assertEqual(len(rows), 2)
        self.assertGreater(summary["silhouette_iou"]["gap_vs_control"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
