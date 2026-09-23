# KUSHULAN Papercut B-LoRA

A research project exploring **style–content separation for Ku Shulan papercut generation with SDXL and B-LoRA**. The workflow connects folk-art data preparation, content and style adaptation, and evaluation of structure, visual style, and decorative symbols.

[中文介绍](../README.md) · [Dataset](DATASET.md) · [Experiments](EXPERIMENTS.md) · [Download and restoration](MIGRATION.md)

## Research workflow

[![Research workflow: dataset construction, style–content decoupled generation, and evaluation](assets/research-pipeline.jpg)](assets/research-pipeline.jpg)

*Project-supplied overview. Numerical claims are preserved as shown in the figure; the evidence available in this repository and remaining verification gaps are documented in [Experiments](EXPERIMENTS.md).*

1. **Data preparation:** organize representative papercut elements and pattern symbols, preserving image versions, processing notes, and caption documents. The figure describes line-art extraction, thickening, recoloring, and annotation.
2. **Generation:** combine the content block from one B-LoRA experiment with the style block from another, using SDXL as the backbone. Existing adapters and the upstream inference entry point are included.
3. **Evaluation design:** assess structural similarity, image/text alignment, and expert judgments of elements, style, and symbols. Metric scripts and per-sample ratings are not yet included; the figure's scores are not independently reproduced here.

## Available assets

The [result gallery](../README.md#生成结果) now shows three content/style/output triplets from the **“7.5 results”** section of the project's Yuque research notes: fish, bird, and a symmetric motif. These are the first three image rows, using the first output in each row. Original images are stored in the repository; no Yuque login is needed to view them here. See [result provenance and comparison notes](RESULTS.md). The original eight cat outputs remain in `Experiment/test-round1/res/`; the online examples have not yet been mapped to individual local checkpoints.

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
