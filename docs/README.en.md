# Ku Shulan colour paste-up papercut design generation

[中文介绍](../README.md)

Ku Shulan (1920–2004) made colour paste-up papercuts, part of the Xunyi colour paste-up papercut listed as national intangible cultural heritage of China. She cut a few flat, saturated papers into pieces and pasted them onto white board. This project studies how an image generator can learn her two craft decisions: which papers to use, and how to divide a figure into pieces that can be cut and pasted by hand.

A paper is in preparation; a link will be added once it is published.

## Method in brief

1. **Dataset.** Colour-calibrated redrawings of her works, grouped as figures, animals, plants, daily objects, window flowers and borders, and split into training, development and test sets.
2. **Paper stock and cut lines.** A paper-stock palette is estimated from the training works. Each work is mapped to its nearest papers and small specks are merged; the boundaries between pieces are the cut lines, at a fine and a coarse level.
3. **Generation model.** A LoRA on the image-editing model Qwen-Image-Edit is trained on pairs of cut lines and works. At use, a line drawing is given directly as the input image: no ControlNet and no per-drawing training.
4. **Evaluation.** Structure, colour and style are compared with other generation methods.

## Repository layout

```text
papercraft/             all code: data preparation, palette, cut lines, training, sampling, evaluation, figures
papercraft/qwen/        editing model: pair preparation, LoRA training, sampling, protocol fixed before testing
papercraft/baselines/   baseline methods
results/cutcraft/       palette, data split, near-duplicate groups and per-image metric tables
Experiment/             earlier B-LoRA weights and checkpoints; cutcraft/ holds the SDXL cut-line style block
docs/                   records of earlier experiments
vendor/B-LoRA/          pinned official B-LoRA implementation
```

## Reproduction

Training and sampling ran on a 96 GB GPU (RTX PRO 6000) in bf16 at 1024 px; LoRA training uses DiffSynth-Studio @7539a33. The images of Ku Shulan's works are not public (see below), so the pipeline needs the data first.

```bash
cd papercraft
python build_data.py                                   # split, paper-stock palette, corpus statistics
python cutlines.py                                     # fine and coarse cut lines
python make_contents.py                                # test drawings
python qwen/prepare_edit_data.py --out ../outputs/qwen_pairs            # training pairs
bash qwen/train_qwen_lora.sh ../outputs/qwen_pairs ../outputs/qwen_work 3
python qwen/qwen_sample.py --lora <LoRA of the selected epoch> --out ../outputs/gen/qwen_q1
python evaluate.py --gen ../outputs/gen/qwen_q1        # structure, colour and style measures
python revision_metrics.py --embed qwen_q1
python stats.py                                        # paired statistics
```

The epoch is chosen on the development drawings by a rule fixed in advance: [papercraft/qwen/PROTOCOL.md](../papercraft/qwen/PROTOCOL.md).

## Data and licence

- The reference images are redrawings of Ku Shulan's works made after images in published books and articles. Her works remain under copyright, and the rights of her heirs and of the redrawers have not been cleared for publication. The images, the test outlines and all image material derived from them (cut-line maps and the like) are therefore **not publicly released**. They are available for non-commercial research on request, subject to the permission of the rights holders.
- The palette, data split, near-duplicate groups and per-image metric tables are in `results/cutcraft/`.
- Code and original documentation are under the [MIT License](../LICENSE); third-party code in `vendor/` keeps its own licence.

## Earlier phase: B-LoRA style–content separation

The repository started as a record of style–content separation with SDXL and B-LoRA. Its weights, checkpoints and records are kept:

- [Experiments](EXPERIMENTS.md) · [Results](RESULTS.md) · [Dataset notes](DATASET.md) · [Migration and verification](MIGRATION.md)
