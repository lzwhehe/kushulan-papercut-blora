"""Leakage-aware train/val/test split over metadata/records.jsonl.

Records are merged into groups when they are byte-identical (same SHA-256) or
perceptually near-identical (768-bit colour dHash of the trimmed image,
Hamming distance <= --max-distance, similar aspect ratio). The default of 120
was calibrated by eye on this data: confirmed duplicates lie at <= 85, the
closest distinct works at >= 198.
Whole groups are then assigned to splits, stratified by category, so that a
duplicate or backup never appears on both sides of a split.

    python tools/split.py                      # needs Pillow for near-duplicates
    python tools/split.py --exact-only         # stdlib only (SHA-256 groups)
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def phash_bits(path, n=16):
    """768-bit colour difference hash of the whitespace-trimmed image."""
    from PIL import Image, ImageChops  # optional dependency
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    mask = ImageChops.difference(im, Image.new("RGB", im.size, "white")).convert("L").point(lambda v: 255 * (v > 24))
    im = im.crop(mask.getbbox() or (0, 0, *im.size))
    bits = 0
    for ch in im.split():
        g = ch.resize((n + 1, n), Image.LANCZOS).tobytes()
        for r in range(n):
            for c in range(n):
                bits = (bits << 1) | (g[r * (n + 1) + c] > g[r * (n + 1) + c + 1])
    return bits, im.width / im.height


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def group(records, hashes=None, max_distance=120, max_aspect=1.15):
    """Return one group id per record; `hashes` holds (bits, aspect) pairs from phash_bits."""
    dsu = DSU(len(records))
    first = {}
    for i, r in enumerate(records):
        if r["sha256"] in first:
            dsu.union(i, first[r["sha256"]])
        first.setdefault(r["sha256"], i)
    if hashes is not None:
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                (hi, ai), (hj, aj) = hashes[i], hashes[j]
                if max(ai, aj) / min(ai, aj) <= max_aspect and bin(hi ^ hj).count("1") <= max_distance:
                    dsu.union(i, j)
    roots = {}
    ids = []
    for i in range(len(records)):
        root = dsu.find(i)
        roots.setdefault(root, f"g{len(roots):04d}")
        ids.append(roots[root])
    return ids


def assign(records, groups, ratios=(0.8, 0.1, 0.1), seed=0):
    """Stratify by the category of each group's first record; returns split per record."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_cat = defaultdict(list)
    for g, idx in members.items():
        by_cat[records[idx[0]]["category"]].append(g)
    rng = random.Random(seed)
    split_of = {}
    for cat in sorted(by_cat):
        gs = sorted(by_cat[cat])
        rng.shuffle(gs)
        n = len(gs)
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        n_val = max(1, round(n * ratios[1])) if n >= 5 else 0
        for k, g in enumerate(gs):
            split_of[g] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return [split_of[g] for g in groups]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=ROOT / "metadata/records.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "metadata/splits.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-distance", type=int, default=120)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    hashes = None if args.exact_only else [phash_bits(ROOT / r["image"]) for r in records]
    groups = group(records, hashes, args.max_distance)
    splits = assign(records, groups, seed=args.seed)
    out = {
        "method": "sha256" if args.exact_only else f"sha256 + colour dHash768 (hamming <= {args.max_distance})",
        "seed": args.seed,
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "counts": {s: splits.count(s) for s in ("train", "val", "test")},
        "groups": len(set(groups)),
        "multi_record_groups": sorted(
            [[r["id"] for r, g2 in zip(records, groups) if g2 == g] for g in set(groups) if groups.count(g) > 1]),
        "assignments": {r["id"]: {"group": g, "split": s} for r, g, s in zip(records, groups, splits)},
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(records)} records -> {out['groups']} groups; {out['counts']}; "
          f"{len(out['multi_record_groups'])} multi-record groups; wrote {args.out}")
