"""Create or verify a SHA-256 manifest of the original research files."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive import digest

ROOTS = ("data", "Experiment", "KUSHULAN_dataset")


def relevant(path):
    return (path.is_file() and not path.name.startswith(("._", "~$"))
            and path.name.lower() not in {".ds_store", "thumbs.db", "desktop.ini"})


def weight_summary(path):
    # Read only JSON metadata. Never unpickle optimizer/random-state files.
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Safetensors header exceeds inspection limit")
        header = json.loads(f.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return {"tensors": len(tensors), "blocks": dict(Counter(
        ".".join(k.split(".")[:5]) for k in tensors)),
        "down_ranks": sorted({v["shape"][0] for k, v in tensors.items()
                              if k.endswith("lora.down.weight")})}


def build(root):
    records = []
    for directory in ROOTS:
        for path in sorted((root / directory).rglob("*")):
            if relevant(path):
                records.append({"path": path.relative_to(root).as_posix(),
                                "bytes": path.stat().st_size, "sha256": digest(path)})
    return records


def verify(root, records):
    errors = []
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"unsafe path: {record['path']}")
        elif not path.is_file():
            errors.append(f"missing: {record['path']}")
        elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {record['path']}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("metadata/source_manifest.jsonl"))
    args = parser.parse_args()
    if args.command == "build":
        records = build(args.root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        summary = {"source_files": len(records), "source_bytes": sum(r["bytes"] for r in records),
                   "categories": {}, "weights": {}}
        for category in sorted((args.root / "KUSHULAN_dataset").iterdir()):
            if category.is_dir():
                summary["categories"][category.name] = sum(relevant(p) and p.suffix.lower() in {".jpg", ".png", ".jpeg"} for p in category.iterdir())
        for path in sorted((args.root / "Experiment").glob("*/*.safetensors")):
            summary["weights"][path.relative_to(args.root).as_posix()] = weight_summary(path)
        (args.manifest.parent / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        errors = verify(args.root, records)
        for error in errors:
            print(error)
        print(f"Checked {len(records)} files; {len(errors)} failures")
        raise SystemExit(bool(errors))
