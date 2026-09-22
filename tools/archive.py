"""Lossless, bounded-memory archive splitting and verified restoration (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split(source, destination, part_size=1024**3):
    source, destination = Path(source), Path(destination)
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    index = destination / "manifest.json"
    if index.exists() or list(destination.glob("archive.part*")):
        raise FileExistsError("Archive destination is not empty")
    parts, total = [], hashlib.sha256()
    with source.open("rb") as src:
        number = 0
        while True:
            block = src.read(min(8 * 1024 * 1024, part_size))
            if not block:
                break
            target = destination / f"archive.part{number:03d}"
            size, checksum = 0, hashlib.sha256()
            with target.open("xb") as dst:
                while block:
                    dst.write(block)
                    checksum.update(block)
                    total.update(block)
                    size += len(block)
                    if size == part_size:
                        break
                    block = src.read(min(8 * 1024 * 1024, part_size - size))
            parts.append({"path": target.name, "bytes": size, "sha256": checksum.hexdigest()})
            number += 1
    record = {"filename": source.name, "bytes": source.stat().st_size,
              "sha256": total.hexdigest(), "parts": parts}
    index.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def restore(manifest, output):
    manifest, output = Path(manifest), Path(output)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if output.exists():
        if output.stat().st_size == record["bytes"] and digest(output) == record["sha256"]:
            return "already verified"
        raise FileExistsError(f"Refusing to overwrite {output}")
    paths = []
    for part in record["parts"]:
        path = (manifest.parent / part["path"]).resolve()
        if path.parent != manifest.parent.resolve():
            raise ValueError("Part path escapes archive directory")
        if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
            raise ValueError(f"Part checksum mismatch: {path.name}")
        paths.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".restoring")
    total = hashlib.sha256()
    with temp.open("xb") as dst:
        for path in paths:
            with path.open("rb") as src:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    total.update(block)
                    dst.write(block)
    if temp.stat().st_size != record["bytes"] or total.hexdigest() != record["sha256"]:
        temp.unlink()
        raise ValueError("Restored archive checksum mismatch")
    temp.rename(output)
    return "restored and verified"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("split")
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p = commands.add_parser("restore")
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "split":
        print(json.dumps(split(args.source, args.destination), ensure_ascii=False, indent=2))
    else:
        print(restore(args.manifest, args.output))
