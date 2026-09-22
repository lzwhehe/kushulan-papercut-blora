# KUSHULAN Papercut B-LoRA

A faithful archive of a Chinese folk papercut research project: image collections, caption documents, two B-LoRA experiment runs, checkpoints, and eight existing generated images.

The curated directories contain 179 representative elements across six categories and 103 pattern images across 21 categories. Both exported adapters contain 320 tensors with rank 64, targeting the content and style blocks used by the official B-LoRA implementation.

The migration preserves 3,443 original research files (9,948,489,017 bytes), excluding operating-system metadata and temporary lock files. Large assets use Git LFS. One 3.49 GB ZIP is stored as four lossless parts with SHA-256 verification.

```bash
git lfs install
git -c lfs.fetchexclude="*" clone https://github.com/lzwhehe/kushulan-papercut-blora.git
cd kushulan-papercut-blora
python -m unittest discover -s tests -v
git lfs pull --include="*" --exclude=""
python tools/archive.py restore archives/manifest.json "data/整理的数据-常用_副本/库淑兰剪纸数据采集/1.库淑兰剪纸作品.zip"
python tools/inventory.py verify
```

The original local files did not include training code or execution logs. A pinned copy of the official B-LoRA training and inference implementation is included under `vendor/B-LoRA`, with upstream attribution and its original license. It was added during migration and is not represented as recovered experiment code. No GPU training or inference was rerun during migration.

The reference diagram's quantitative evaluation scores are not presented as verified results. Likewise, 282 curated images do not establish 282 fully paired line-art/color records. The source ZIP contains 1,153 image entries, which has not been reconciled with the diagram's 2,008 artworks.

See the [main README](../README.md) for the original result gallery, inference commands, dataset structure, and evidence boundaries. The root MIT license covers new migration tools and original documentation only; datasets, artwork, documents, and pre-existing model weights are not relicensed.
