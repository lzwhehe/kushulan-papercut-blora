# KUSHULAN Papercut B-LoRA

### 库淑兰剪纸的风格与内容分离生成研究

[![Migration tools](https://github.com/lzwhehe/kushulan-papercut-blora/actions/workflows/ci.yml/badge.svg)](https://github.com/lzwhehe/kushulan-papercut-blora/actions/workflows/ci.yml)
[![SDXL](https://img.shields.io/badge/Backbone-SDXL-5865F2)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![B-LoRA](https://img.shields.io/badge/Method-B--LoRA-BD3B36)](https://github.com/yardenfren1996/B-LoRA)

本项目以库淑兰剪纸为研究对象，探索如何在保留人物、动物、植物等元素结构的同时，迁移剪纸的色彩、纹样与装饰风格。研究流程围绕 **数据整理 → SDXL / B-LoRA 风格与内容分离 → 生成与评估** 展开，将传统剪纸资料与生成式模型实验连接起来。

仓库提供 **179 张代表元素、103 张纹样符号、两轮 LoRA 实验权重与检查点，以及内容—风格—生成结果对照**，并附带提示词资料、官方训练/推理入口和逐文件校验工具，便于查看研究过程与继续实验。

[English](docs/README.en.md) · [方法流程](#方法流程) · [生成结果](#生成结果) · [快速开始](#快速开始) · [数据说明](docs/DATASET.md) · [实验记录](docs/EXPERIMENTS.md) · [完整下载](docs/MIGRATION.md)

## 方法流程

[![库淑兰剪纸研究流程：半结构化数据构建、风格与内容分离生成、自动指标与专家评价](docs/assets/research-pipeline.jpg)](docs/assets/research-pipeline.jpg)

*研究流程概览，点击图片可查看原图。图中数值保留所提供流程图的原始口径；与当前仓库资料的对应关系和待复核项见 [实验记录](docs/EXPERIMENTS.md#与参考图的对应关系)。*

### 1. 剪纸元素与纹样整理

从原始作品中整理代表元素与纹样符号，按人物、动物、植物、日常器物、窗花和边框组织数据。流程图进一步描述了线稿提取、轮廓加粗、人工重着色和语义标注方案；仓库保留各阶段已有图像版本、处理文档与提示词资料。

### 2. 风格与内容分离生成

以 SDXL 为基础，使用 B-LoRA 的目标注意力块分别提取内容与风格信息，在推理时组合来自不同实验的权重。线稿与内容描述用于表达元素结构，风格参考与包含纹样语义的提示词用于描述色彩和装饰特征。现有两份导出权重均包含内容块与风格块，可通过官方推理入口选择组合。

### 3. 结构、风格与符号表达评估

流程图中的评估设计结合 Edge F1、silhouette IoU、CLIP-I / CLIP-T，以及结构、元素、风格和符号维度的专家评分。当前仓库保存了生成样例；配对评估数据、评分表与指标复现脚本尚待补充，图中分数不作为本仓库已复跑验证的结论。

## 生成结果

以下选自研究记录 [《新结果与7.2测试结果对比》](https://www.yuque.com/ariel-cgurv/br3z17/sba8ooeg1ir3fpaz) 的 **“7.5结果”**：从左到右依次为内容线稿、风格参考和生成图。展示原表前三组，每组采用原表的第一张结果，便于观察形态、配色与装饰纹样之间的关系。

| 内容线稿 · Content | 风格参考 · Style | 生成结果 · Output |
| :---: | :---: | :---: |
| <img src="docs/assets/results/fish-content.png" width="240" alt="鱼形内容线稿"> | <img src="docs/assets/results/fish-style.png" width="240" alt="猫形剪纸风格参考"> | <img src="docs/assets/results/fish-result.jpeg" width="240" alt="7.5实验的鱼形生成结果"> |
| <img src="docs/assets/results/bird-content.png" width="240" alt="展翅鸟形内容线稿"> | <img src="docs/assets/results/bird-style.jpeg" width="240" alt="长尾动物剪纸风格参考"> | <img src="docs/assets/results/bird-result.jpeg" width="240" alt="7.5实验的鸟形生成结果"> |
| <img src="docs/assets/results/motif-content.png" width="240" alt="左右对称纹样内容线稿"> | <img src="docs/assets/results/motif-style.jpeg" width="240" alt="蜘蛛形剪纸风格参考"> | <img src="docs/assets/results/motif-result.jpeg" width="240" alt="7.5实验的对称纹样生成结果"> |

这些是研究文档中的既有实验图片，原图已保存到仓库，未重新生成或修饰。图像与当前两份本地权重的逐次运行对应关系尚待补全；原始 8 张猫图仍保留在 `Experiment/test-round1/res/`。更多对照、提示词与轮廓线实验的资料说明见 [生成结果与对比记录](docs/RESULTS.md)。

## 已有数据与实验资产

| 内容 | 本地核实结果 |
| --- | --- |
| 代表元素 | 179 张，人物 30、动物 46、日常器物 25、植物 38、窗花 20、边框 20 |
| 纹样符号 | 103 张，21 个目录类别 |
| 整理后图像总数 | 282 张；代表元素与纹样目录合计，不表示已验证的线稿/彩色配对记录 |
| 原始作品 ZIP | 1,153 个图像条目；与参考图的 2,008 件原作口径尚未对应 |
| 实验 | 两轮，各有 checkpoint-500 和 checkpoint-1000 |
| 导出权重 | `ksl_style.safetensors`、`ksl_content.safetensors` |
| 权重结构 | 每份 320 个 tensor，LoRA rank 64，覆盖两个目标注意力块 |
| 生成结果展示 | 语雀“7.5结果”中的 3 组内容/风格/结果对照；第一轮目录另保留原始 8 张猫图 |
| 迁移清单 | 3,443 个有效源文件，9,948,489,017 bytes，逐文件 SHA-256 |

**复现状态：** 数据、权重与检查点已归档并校验。原始训练脚本与运行日志未在源资料中找到；`vendor/B-LoRA/` 为后续补充的固定版本官方实现。本次迁移未重新执行 GPU 训练或推理。训练配方、历史参数与评估结果的可复核范围见 [实验记录](docs/EXPERIMENTS.md)。

## 快速开始

查看代码、说明和生成样例，不下载全部数据：

```bash
git lfs install
git -c lfs.fetchexclude="*" clone https://github.com/lzwhehe/kushulan-papercut-blora.git
cd kushulan-papercut-blora
python -m unittest discover -s tests -v
```

完整迁移资料约 9.27 GiB；大文件存放在 Git LFS。仅下载主要数据和最终权重：

```bash
git lfs pull --include="KUSHULAN_dataset/**,Experiment/test-round1/ksl_style.safetensors,Experiment/test-round2/ksl_content.safetensors" --exclude=""
```

下载全部资产、还原分块压缩包并核对原始文件：

```bash
git lfs pull --include="*" --exclude=""
python tools/archive.py restore archives/manifest.json "data/整理的数据-常用_副本/库淑兰剪纸数据采集/1.库淑兰剪纸作品.zip"
python tools/inventory.py verify
```

不要只通过 GitHub 的源码 ZIP 判断数据是否齐全：LFS 文件可能仍是文本指针。

## 使用现有权重推理

训练和推理需要单独的 Python 环境、CUDA GPU 与 SDXL 模型。上游使用旧版 Diffusers 0.25.0，环境与现代 PEFT 版 LoRA 流程不能直接混用。以下是官方代码入口；本次只验证了资产完整性和迁移工具，没有验证 GPU 兼容性。

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r vendor/B-LoRA/requirements.txt
python -c "from pathlib import Path; Path('outputs/demo').mkdir(parents=True, exist_ok=True)"
python vendor/B-LoRA/inference.py --prompt="A cat in ksl style" --content_B_LoRA="Experiment/test-round2/ksl_content.safetensors" --style_B_LoRA="Experiment/test-round1/ksl_style.safetensors" --output_path="outputs/demo"
```

该命令使用第二轮权重的 `unet.up_blocks.0.attentions.0` 内容块，以及第一轮权重的 `unet.up_blocks.0.attentions.1` 风格块。文件名中的 `content` / `style` 表示实验用途；两份文件实际上都包含两个块。它们不是两个只能分别表示内容或风格的单块文件。

更多训练说明、参数证据和现有限制见 [实验记录](docs/EXPERIMENTS.md)。

## 目录

```text
data/                   原始整理资料、处理后元素/纹样、提示词 DOCX、ZIP
KUSHULAN_dataset/       六类编号代表元素，保留原始文件名
Experiment/            两轮导出权重、训练检查点和第一轮生成结果
archives/              大型 ZIP 的四个无损分块与校验清单
metadata/              原始文件 SHA-256 清单与汇总
vendor/B-LoRA/         固定版本的上游训练/推理代码及原许可证
tools/                  分块、还原、清单构建与验证工具
tests/                  迁移工具的单元测试
docs/                   研究流程图、数据、实验和迁移说明
```

原始文件名中的 `.jpg.jpg`、中文目录及重复备份均予以保留，以避免破坏已有提示词与实验资料的对应关系。仅排除 `.DS_Store`、AppleDouble `._*` 和 Office `~$*` 等系统/临时文件。

## 来源与许可

本项目使用 [B-LoRA 官方实现](https://github.com/yardenfren1996/B-LoRA)，方法来自 Frenkel 等人的 [Implicit Style-Content Separation using B-LoRA](https://arxiv.org/abs/2403.14572)。本仓库的贡献是剪纸场景资料整理、已有实验资产保存与迁移工程；B-LoRA 方法和官方代码归原作者所有。

新增迁移工具与原创说明采用 [MIT License](LICENSE)。第三方代码保留其原许可证。研究流程图、剪纸作品、原始文档、数据与模型权重不自动适用代码的 MIT 许可；目前未在原始资料中找到单独的数据授权文件，见 [数据说明](docs/DATASET.md)。
