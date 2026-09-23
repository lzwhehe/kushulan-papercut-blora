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

README 的推理命令直接调用未修改的官方入口。该入口不提供 seed 参数，也不输出完整运行配置，不能用它声称严格重现某张历史生成图。输出目录须先创建。默认会获取 SDXL 与上游指定的 VAE 模型。

## 与参考图的对应关系

| 流程图内容 | 当前证据 |
| --- | --- |
| 六类元素、21 类纹样 | 文件目录和计数可验证 |
| 282 条半结构化记录 | 282 张整理后图像可验证，完整配对记录待补 |
| GPT-4o 提取、Qwen-VL 检查、ΔE 阈值 | 未发现可复核 API 日志或测量记录 |
| 25 张线稿、5 张风格参考训练配方 | 未发现样本清单，待补 |
| SDXL B-LoRA 内容/风格块 | 权重模块结构可验证 |
| 50 步、CFG 5、1024 分辨率 | 参考图设定；原始生成参数未保存 |
| Edge F1、silhouette IoU、CLIP-I / CLIP-T | 未发现评估脚本、配对清单或结果表 |
| 专家均分、Kendall's W | 未发现逐样本评分与评审记录 |

后续补充历史代码与记录时，应新增可追溯文件与配置，不覆盖原始实验。定量比较应固定种子、提示词、参考图、数据划分和指标实现版本，再发布结果。
