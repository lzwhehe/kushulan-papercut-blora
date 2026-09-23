"""Build the verifiable record table (image <-> caption <-> category) from the DOCX prompts.

Each DOCX under data/第二次_副本/*提示词/ holds `{"file_name": "N.jpg", "text": ...}`
paragraphs. Image files in the sorted folders use mixed extensions
(`N.jpg`, `N.jpg.jpg`, `N.jpg.png`, `N.png`), so records are matched on the
numeric stem, and every match is reported. Byte-identical copies in other
dataset folders are found through metadata/source_manifest.jsonl.

    python tools/build_records.py            # writes metadata/records.jsonl
    python tools/build_records.py --check    # fails if the committed table is stale
"""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "data/第二次_副本"
SETS = {
    "element": (f"{BASE}/库淑兰剪纸代表提示词", f"{BASE}/库淑兰剪纸代表元素（排序后）"),
    "symbol": (f"{BASE}/纹样符号提示词", f"{BASE}/纹样符号（排序后）"),
}
CATEGORY_EN = {
    "人物": "Figure", "动物": "Animal", "日常": "Object", "植物": "Plant", "窗花": "Window flower", "边框": "Border",
    "圆点纹": "Dot", "壑牙子纹": "Heya", "太阳纹": "Sun", "头饰纹": "HeadOrnament", "挂钱纹": "HangingCoin",
    "方胜如意纹": "SquareAuspiciousCloud", "月牙纹": "Crescent", "杂树纹": "Tree", "桃儿纹": "Peach",
    "梅花纹": "Plum", "水纹": "Water", "牡丹纹": "Peony", "莲花纹": "Lotus", "菊花纹": "Chrysanthemum",
    "蜂蝶纹": "Butterfly", "钱串子纹": "StringofCoins", "锯齿纹": "Glands", "门帘纹": "Curtain", "风纹": "Wind",
    "鱼纹": "Fish", "鸟纹": "Bird",
}
# DOCX names that differ from their image folder in the source material.
DOCX_ALIAS = {"头饰": "头饰纹", "挂线纹": "挂钱纹", "圆点纹": "圆点纹"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

# Appearance vocabulary used to split captions into content vs. style/symbol terms.
STYLE_WORDS = re.compile(
    r"(red|yellow|blue|green|pink|black|white|orange|purple|colou?r|background|accent|palette|bright|vivid|"
    r"flat_design|lineart|linework|pattern|motif|decorat|ornament|layered|nested|repeating|stripe|dots?|"
    r"sunburst|floral|symmetr|radial|interlocking|texture|embellish|combination|navy|golden|tone)",
    re.I)
RECORD = re.compile(r'"file_name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"text"\s*:\s*"?(?P<text>.*?)"?\s*}?\s*$', re.S)
LEAD = re.compile(r"^.*?(composed of|made up of) geometric figures\s*\.?\s*", re.I)


def docx_paragraphs(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = html.unescape(re.sub(r"<[^>]+>", "", para)).strip()
        if text:
            yield text


def parse_docx(path):
    """Return [(file_name, caption)] preserving document order."""
    records = []
    for para in docx_paragraphs(path):
        m = RECORD.search(para)
        if m:
            text = re.sub(r"\s+", " ", m["text"].replace("，", ", ")).strip().strip('"').strip()
            records.append((m["name"].strip(), text))
    return records


def split_terms(caption):
    """Split a caption into content and style/symbol terms (heuristic, documented in DATASET.md)."""
    body = LEAD.sub("", caption)
    if body.count("_") >= 3:  # tag list: "animal, cat, round_body, red_accents"
        chunks = [t.strip(" .") for t in body.split(",")]
    else:  # prose: split into clauses
        chunks = [c.strip(" .") for c in re.split(r"[,.;]| and ", body)]
    chunks = [c for c in chunks if c]
    content = [c for c in chunks if not STYLE_WORDS.search(c)]
    style = [c for c in chunks if STYLE_WORDS.search(c)]
    return content, style


def stem_number(name):
    m = re.search(r"(\d+)", name)
    return int(m[1]) if m else None


def load_manifest(root):
    path = root / "metadata/source_manifest.jsonl"
    by_path, by_hash = {}, defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_path[rec["path"]] = rec
            by_hash[rec["sha256"]].append(rec["path"])
    return by_path, by_hash


def build(root=ROOT):
    by_path, by_hash = load_manifest(root)
    records, problems = [], []
    for set_name, (doc_dir, img_dir) in SETS.items():
        for docx in sorted((root / doc_dir).glob("*.docx")):
            if docx.name.startswith("~$"):
                continue
            cat_zh = docx.stem.removesuffix("提示词")
            cat_zh = DOCX_ALIAS.get(cat_zh, cat_zh)
            folder = root / img_dir / f"{cat_zh}.jpg"
            if not folder.is_dir():
                problems.append(f"no image folder for {docx.name}: {folder.name}")
                continue
            images = {}
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXT and not p.name.startswith("._"):
                    n = stem_number(p.name)
                    if n in images:
                        problems.append(f"duplicate index {n} in {folder.name}: {images[n].name}, {p.name}")
                    images[n] = p
            seen = set()
            for name, caption in parse_docx(docx):
                n = stem_number(name)
                image = images.get(n)
                if image is None:
                    problems.append(f"{docx.name}: {name} has no image")
                    continue
                seen.add(n)
                rel = image.relative_to(root).as_posix()
                man = by_path.get(rel)
                if man is None:
                    problems.append(f"{rel} not in source manifest")
                    continue
                content, style = split_terms(caption)
                records.append({
                    "id": f"{set_name}/{CATEGORY_EN[cat_zh]}/{n:03d}",
                    "set": set_name,
                    "category": CATEGORY_EN[cat_zh],
                    "category_zh": cat_zh,
                    "index": n,
                    "image": rel,
                    "sha256": man["sha256"],
                    "bytes": man["bytes"],
                    "copies": sorted(p for p in by_hash[man["sha256"]] if p != rel),
                    "caption": caption,
                    "content_terms": content,
                    "style_terms": style,
                    "line_art": None,  # no paired line-art file exists in the archived data
                    "caption_source": docx.relative_to(root).as_posix(),
                })
            for n in sorted(set(images) - seen):
                problems.append(f"{images[n].relative_to(root).as_posix()} has no caption")
    return records, problems


def dumps(records):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "metadata/records.jsonl")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records, problems = build(args.root)
    for p in problems:
        print("WARN", p, file=sys.stderr)
    text = dumps(records)
    counts = defaultdict(int)
    for r in records:
        counts[r["set"]] += 1
    summary = f"{len(records)} records ({', '.join(f'{k} {v}' for k, v in counts.items())}); {len(problems)} problems"
    if args.check:
        ok = args.out.is_file() and args.out.read_text(encoding="utf-8") == text
        print(summary + ("; table up to date" if ok else "; table STALE"))
        raise SystemExit(0 if ok and not problems else 1)
    args.out.write_text(text, encoding="utf-8")
    print(summary + f"; wrote {args.out}")
