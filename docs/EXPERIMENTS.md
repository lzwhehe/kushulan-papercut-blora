# 实验记录与运行入口

## 补充的在线研究记录

README 的生成结果已更新为语雀“7.5结果”中的三组内容线稿、风格参考与输出图。另有提示词复杂度和轮廓线粗细的对比资料，见 [生成结果与对比记录](RESULTS.md)。这些资料补充了视觉样例与实验思路，但尚未建立在线图片与下述本地检查点的逐次运行映射。

## 原始产物

| 实验 | 导出文件 | 检查点 | 已有生成图 |
| --- | --- | --- | ---: |
| test-round1 | `ksl_style.safetensors` | 500、1000 | 8 |
| test-round2 | `ksl_content.safetensors` | 500、1000 | 0 |

每个检查点保留 `pytorch_lora_weights.safetensors`、`optimizer.bin`、`scheduler.bin`、`scaler.pt` 和 `random_states_0.pkl`。清单工具只读取 safetensors 的 JSON 头，不执行 pickle 或反序列化训练状态。

两份导出权重均有 320 个 tensor，LoRA down 矩阵显示 rank 64：

| 模块前缀 | tensor 数 | 官方方法中的用途 |
| --- | ---: | --- |
| `unet.up_blocks.0.attentions.0` | 160 | 内容 |
| `unet.up_blocks.0.attentions.1` | 160 | 风格 |

这与官方 B-LoRA 的目标模块对应。仅凭结构不能证明具体训练图像、训练配方或质量指标。

## 官方入口

本地源资料中没有 `.py`、`.ipynb`、训练配置或日志。为便于后续使用，引入官方 B-LoRA 固定提交的训练与推理代码；来源和逐文件哈希见 `vendor/B-LoRA/PROVENANCE.md` 和 `vendor/B-LoRA/source.json`。

环境依赖保留上游 `requirements.txt`，包含 Diffusers 0.25.0。部分依赖未被上游锁定，不能据此保证在所有现代系统中直接安装成功。建议在独立的 Linux CUDA / Python 3.11 环境中安装，并在成功复跑后保存完整 `pip freeze`、GPU 和 CUDA 信息。

训练命令模板（Linux shell）：

```bash
accelerate launch vendor/B-LoRA/train_dreambooth_b-lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="PATH_TO_SELECTED_TRAINING_IMAGES" \
  --output_dir="outputs/new-training-run" \
  --instance_prompt="A [v]" \
  --resolution=1024 --rank=64 --train_batch_size=1 \
  --learning_rate=5e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=1000 --checkpointing_steps=500 --seed=0 \
  --gradient_checkpointing --use_8bit_adam --mixed_precision="fp16"
```

这些数值来自上游示例，是**新的运行模板**，不是恢复出来的原实验配置。使用前替换图像目录和提示词；原有多样本数据与上游单图 B-LoRA 实验设定也需要分别记录。

需要可复现的生成时，请使用 `tools/generate.py`：它调用同一套 `blora_utils`，但固定种子、步数、CFG 与分辨率，并在 `runs/<name>/run.json` 中记录参数、权重 SHA-256、依赖版本、GPU 和 git 提交；同名目录不会被覆盖。`--dry_run` 可在无 GPU 时检查任务表。

README 的推理命令直接调用未修改的官方入口。该入口不提供 seed 参数，也不输出完整运行配置，不能用它声称严格重现某张历史生成图。输出目录须先创建。默认会获取 SDXL 与上游指定的 VAE 模型。

## 与参考图的对应关系

| 流程图内容 | 当前证据 |
| --- | --- |
| 六类元素、21 类纹样 | 文件目录和计数可验证 |
| 282 条半结构化记录 | 282 条“图像 + 描述 + 类别”记录已验证（`metadata/records.jsonl`）；其中 270 张唯一；无配对线稿 |
| GPT-4o 提取、Qwen-VL 检查、ΔE 阈值 | 未发现可复核 API 日志或测量记录 |
| 25 张线稿、5 张风格参考训练配方 | 未发现样本清单，待补 |
| SDXL B-LoRA 内容/风格块 | 权重模块结构可验证 |
| 50 步、CFG 5、1024 分辨率 | 参考图设定；原始生成参数未保存 |
| Edge F1、silhouette IoU、CLIP-I / CLIP-T | 已补充实现（`tools/evaluate.py`）；原结果表仍未发现 |
| 专家均分、Kendall's W | 计算脚本已补充（`evaluate.py agreement`）；逐样本评分仍未发现 |

后续补充历史代码与记录时，应新增可追溯文件与配置，不覆盖原始实验。定量比较应固定种子、提示词、参考图、数据划分和指标实现版本，再发布结果。

## 结构指标的有效性检查

B-LoRA 的内容块学习“物体是什么”，不约束像素位置；输出的姿态、大小可以变化，背景常被纹样填满。Edge F1 与 silhouette IoU 却假设输出与线稿逐像素对齐。为检验它们在本流程中是否有效，`tools/evaluate.py` 对每个输出同时计算与**另一组样本线稿**的指标（错配对照），并报告差值的 bootstrap 95% 置信区间。

在已归档的三组 7.5 结果上（512 px，容差 3 px）：

| 指标 | 匹配 | 错配对照 | 差值 [95% CI] |
| --- | ---: | ---: | ---: |
| Edge F1 | 0.230 | 0.259 | −0.028 [−0.110, 0.046] |
| Silhouette IoU | 0.132 | 0.124 | +0.009 [−0.004, 0.034] |

两者都不高于对照。样本只有 3 组，只能说明这两项指标在无空间条件的 B-LoRA 生成上**缺乏区分度**，不能作为结构保真度的证据。建议：

1. 无空间条件时，以 DINOv2 相似度（`--dino`）和专家评分衡量内容保真；
2. 若需要像素级结构保真，加入 ControlNet-lineart 或 img2img 等空间条件，再使用 Edge F1 / IoU，并始终报告对照差值；
3. 用 `tools/generate.py` 固定种子批量生成（例如每组 4 个种子），在 `metadata/splits.json` 的 test 划分上评估。

## 后续实验计划

| 优先级 | 任务 | 所需资源 |
| --- | --- | --- |
| 1 | 用现有两份权重和 `configs/jobs_example.csv` 复跑 4 个种子，建立第一份带 `run.json` 的基线 | 单卡 GPU（≥ 16 GB） |
| 2 | 为 test 划分的元素补齐线稿（与记录 `id` 对应），填入 `line_art` 字段 | 人工 / 线稿提取 |
| 3 | 在 train 划分上按新配方重训内容与风格 B-LoRA，记录完整命令与日志 | GPU |
| 4 | 消融：α_c / α_s 网格；简单与增强风格描述；线稿粗细 | GPU |
| 5 | 对比：B-LoRA vs. 单一 LoRA、IP-Adapter 风格参考、B-LoRA + ControlNet-lineart | GPU |
| 6 | 专家评分：按 `sample,rater,dimension,score` 收集，`evaluate.py agreement` 计算 Kendall's W | 3 名以上评审 |
