# 库淑兰彩贴剪纸设计生成

[English README](docs/README.en.md)

库淑兰（1920–2004）的彩贴剪纸是国家级非物质文化遗产“旬邑彩贴剪纸”的代表：用少数几种平涂、饱和的彩纸剪成纸片，再贴到白纸板上。本项目研究如何让图像生成模型学会她的两项工艺决定：选哪几种纸，以及把一个图形分成哪些剪得出、贴得上的纸片。

相关论文正在准备中，发表后在此补充链接。

## 方法概要

1. **数据集**：库淑兰作品的校色重绘图，按人物、动物、植物、日常器物、窗花、边框分类，分为训练、开发和测试三部分。
2. **纸色与剪切线**：从训练作品中提取一组“纸色谱”；每件作品归到最近的纸色并合并过小的碎块，相邻纸片之间的边界即“剪切线”，分细、粗两级。
3. **生成模型**：用“剪切线 → 作品”的成对数据训练图像编辑模型 Qwen-Image-Edit 的 LoRA。使用时直接输入线稿，不需要 ControlNet，也不需要为每张线稿单独训练。
4. **评价**：从结构、色彩、风格三方面与其他生成方法比较。

## 仓库结构

```text
papercraft/             全部代码：数据准备、色板、剪切线、训练、采样、评价、出图
papercraft/qwen/        编辑模型的数据准备、LoRA 训练、采样脚本，以及测试前登记的实验协议 PROTOCOL.md
papercraft/baselines/   基线方法
results/cutcraft/       色板、数据划分、近重复分组与逐图指标表
Experiment/             早期 B-LoRA 权重与检查点；cutcraft/ 为 SDXL 阶段的剪切线风格块
docs/                   早期实验记录
vendor/B-LoRA/          B-LoRA 官方实现（固定版本）
```

## 复现

训练和采样使用 96 GB 显卡（RTX PRO 6000）、bf16、1024 px；LoRA 训练基于 DiffSynth-Studio @7539a33。库淑兰作品的图像不公开（见下文），因此下面的流程需要先获得数据。

```bash
cd papercraft
python build_data.py                                   # 数据划分、纸色谱、语料统计
python cutlines.py                                     # 细、粗两级剪切线
python make_contents.py                                # 测试线稿
python qwen/prepare_edit_data.py --out ../outputs/qwen_pairs            # 训练数据
bash qwen/train_qwen_lora.sh ../outputs/qwen_pairs ../outputs/qwen_work 3
python qwen/qwen_sample.py --lora <选定 epoch 的 LoRA> --out ../outputs/gen/qwen_q1
python evaluate.py --gen ../outputs/gen/qwen_q1        # 结构、色彩、风格指标
python revision_metrics.py --embed qwen_q1
python stats.py                                        # 配对统计
```

Epoch 只在开发线稿上按事先定好的规则选择，见 [papercraft/qwen/PROTOCOL.md](papercraft/qwen/PROTOCOL.md)。

## 数据与许可

- 参考图是根据已出版书籍和文献中的图片绘制的库淑兰作品重绘图。她的作品仍受著作权保护，其继承人和重绘者的权利尚未取得公开授权，因此作品图像、测试线稿以及由它们派生的图像材料（剪切线图等）**不公开发布**；非商业研究可向作者申请，需经权利人同意。
- 纸色谱、数据划分、近重复分组和逐图指标表公开在 `results/cutcraft/`。
- 代码与原创文档采用 [MIT License](LICENSE)；第三方代码（`vendor/`）保留原许可证。

## 项目早期：B-LoRA 风格与内容分离

本仓库最初记录的是基于 SDXL 和 B-LoRA 的风格与内容分离实验。相关权重、检查点和实验记录仍保留在仓库中：

- [实验记录](docs/EXPERIMENTS.md) · [生成结果与对比](docs/RESULTS.md) · [数据说明](docs/DATASET.md) · [资料迁移与校验](docs/MIGRATION.md)
