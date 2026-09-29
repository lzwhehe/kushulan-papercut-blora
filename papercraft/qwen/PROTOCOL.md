# Qwen-Image-Edit-2511 backbone experiment: protocol (fixed before any test result)

Written 2026-09-29, while the main LoRA was training and before any test drawing was generated.

## Model, data, training
- Qwen/Qwen-Image-Edit-2511 @ 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 (Apache 2.0), bf16, RTX PRO 6000 (96 GB).
- Edit pairs from the same 257 training images and split as the SDXL cut-line block:
  - input = automatically derived cut-line map (dense or coarse), black on white;
  - target = catalogue image;
  - with horizontal flips, 1,028 pairs.
- Instruction: `prepare_edit_data.INSTRUCTION`.
- LoRA rank 32, lr 1e-4, 1024 px, 3 epochs (~3,084 steps), DiffSynth-Studio @7539a33.
- Checkpoint after every epoch.

## Epoch selection (development set only)
- 3 development drawings × 4 seeds (0–3), unguided (Q1), 40 steps, true CFG 4.
- Rule: among epochs with mean dev line recall (3 px, 512 px) ≥ 0.90, take the lowest mean palette distance. If no epoch reaches 0.90, take the highest line recall.
- Faces and other small details are inspected visually and reported, but do not enter the rule.

## Groups on the 34 test drawings (seeds 0 and 1)
| Group | Setting |
|---|---|
| Q0 | base model, instruction only |
| Q1 | + cut-line LoRA (selected epoch) |
| Q2 | Q1 + CraftGuide: η = 1.0, n = 2, window 15–75 %, same energy and weights as SDXL, full-resolution VAE decode |
| Q3 | Q2 + cut projection and enforced 200 mm SVG export (as for SDXL) |
| Q1′ | LoRA trained on Canny edges of the images instead of cut-line maps (same settings) |
| Q1s | LoRA trained on the strict split (235 images) |

The open-stroke (fragment) augmentation is an ablation only, at 30 %, and is reported as such.

## Primary comparisons (drawing-level, Holm-corrected, bootstrap CIs; as in `stats.py`)
- Q2 vs SDXL CutCraft on line recall, silhouette IoU, CLIP style (de-duplicated refs) and colour-mix distance.
- Q1 vs Q0 on the same four endpoints (effect of the cut-line pairs on the new backbone).

## Reporting
- All other comparisons are exploratory.
- Results are reported whether or not the new backbone improves on SDXL.
- Pilot runs on the 20 GB GPU (NF4, 512 px) are not reported as results.
