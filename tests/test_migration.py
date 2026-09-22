import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from archive import split, restore
from inventory import relevant, build, verify


class MigrationTests(unittest.TestCase):
    def test_roundtrip_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "作品.zip"
            source.write_bytes(bytes(range(256)) * 7)
            record = split(source, root / "parts", part_size=300)
            self.assertEqual(len(record["parts"]), 6)
            target = root / "restored.zip"
            restore(root / "parts/manifest.json", target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(restore(root / "parts/manifest.json", target), "already verified")

    def test_corruption_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.write_bytes(b"abcde")
            split(source, root / "parts", 2)
            (root / "parts/archive.part001").write_bytes(b"xx")
            with self.assertRaises(ValueError):
                restore(root / "parts/manifest.json", root / "out")
            self.assertFalse((root / "out").exists())

    def test_different_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.write_bytes(b"abc")
            split(source, root / "parts", 2)
            out = root / "out"
            out.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                restore(root / "parts/manifest.json", out)
            self.assertEqual(out.read_bytes(), b"keep")

    def test_inventory_ignores_metadata_and_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir()
            (root / "data/sample.jpg").write_bytes(b"sample")
            (root / "data/._sample.jpg").write_bytes(b"metadata")
            records = build(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(verify(root, records), [])
            (root / "data/sample.jpg").write_bytes(b"changed")
            self.assertEqual(len(verify(root, records)), 1)

    def test_archive_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "parts").mkdir()
            record = {"bytes": 1, "sha256": "unused", "parts": [
                {"path": "../secret", "bytes": 1, "sha256": "unused"}]}
            manifest = root / "parts/manifest.json"
            manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                restore(manifest, root / "out")


if __name__ == "__main__":
    unittest.main()
