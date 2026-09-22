# KUSHULAN Papercut B-LoRA

基于库淑兰剪纸资料的 **SDXL / B-LoRA 风格与内容分离实验归档**。本仓库完整迁移原项目中的数据、提示词文档、两轮训练检查点和生成样例，并补充可校验的数据清单、官方训练/推理入口和克隆还原说明。

**当前状态：实验资产已保存，原始训练脚本与运行日志未在本地资料中找到。** `vendor/B-LoRA/` 是迁移时引入的官方实现，不冒充原实验代码；尚未在本次迁移中重新进行 GPU 训练或推理。图中的自动评估分数、专家评分和训练样本配方尚未由日志复核。

[English overview](docs/README.en.md) · [完整下载与还原](docs/MIGRATION.md) · [数据说明](docs/DATASET.md) · [实验记录](docs/EXPERIMENTS.md) · [上游来源](vendor/B-LoRA/PROVENANCE.md)

## 已有生成结果

以下图片来自原项目 `Experiment/test-round1/res/`，为既有结果，并非本次迁移新生成。

| A cat in ksl style | A kushulan_cat in kushulan animal collection style |
| --- | --- |
| ![Original cat result](Experiment/test-round1/res/A%20cat%20in%20ksl%20style_0.jpg) | ![Original collection result](Experiment/test-round1/res/A%20kushulan_cat%20in%20kushulan%20animal%20collection%20style_0.jpg) |
| ![Original cat variant](Experiment/test-round1/res/A%20cat%20in%20ksl%20style_1.jpg) | ![Original collection variant](Experiment/test-round1/res/A%20kushulan_cat%20in%20kushulan%20animal%20collection%20style_1.jpg) |

## 项目内容

| 内容 | 本地核实结果 |
| --- | --- |
| 代表元素 | 179 张，人物 30、动物 46、日常器物 25、植物 38、窗花 20、边框 20 |
| 纹样符号 | 103 张，21 个目录类别 |
| 整理后图像总数 | 282 张；代表元素与纹样目录合计，不表示已验证的线稿/彩色配对记录 |
| 原始作品 ZIP | 1,153 个图像条目；与参考图的 2,008 件原作口径尚未对应 |
| 实验 | 两轮，各有 checkpoint-500 和 checkpoint-1000 |
| 导出权重 | `ksl_style.safetensors`、`ksl_content.safetensors` |
| 权重结构 | 每份 320 个 tensor，LoRA rank 64，覆盖两个目标注意力块 |
| 已有生成图 | 8 张，位于第一轮实验目录 |
| 迁移清单 | 3,443 个有效源文件，9,948,489,017 bytes，逐文件 SHA-256 |

```mermaid
flowchart LR
    A[原始剪纸资料与处理文档] --> B[179 个代表元素与 103 个纹样]
    B --> C[图像与提示词整理]
    C -.原实验脚本与日志待补.-> D[两轮 B-LoRA 权重与检查点]
    D --> E[内容块与风格块组合推理]
    E --> F[已有 8 张生成结果]
    F -.待建立配对与标注.-> G[结构指标与专家评估]
```

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
docs/                   数据、实验和迁移说明
```

原始文件名中的 `.jpg.jpg`、中文目录及重复备份均予以保留，以避免破坏已有提示词与实验资料的对应关系。仅排除 `.DS_Store`、AppleDouble `._*` 和 Office `~$*` 等系统/临时文件。

## 来源与许可

本项目使用 [B-LoRA 官方实现](https://github.com/yardenfren1996/B-LoRA)，方法来自 Frenkel 等人的 [Implicit Style-Content Separation using B-LoRA](https://arxiv.org/abs/2403.14572)。本仓库的贡献是剪纸场景资料整理、已有实验资产保存与迁移工程；B-LoRA 方法和官方代码归原作者所有。

新增迁移工具与原创说明采用 [MIT License](LICENSE)。第三方代码保留其原许可证。剪纸作品、原始文档、数据与模型权重不自动适用代码的 MIT 许可；目前未在原始资料中找到单独的数据授权文件，见 [数据说明](docs/DATASET.md)。
