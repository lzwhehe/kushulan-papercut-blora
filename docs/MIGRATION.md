# 迁移与还原

本次源目录为 `E:\P_KUSHULAN`。迁移不改写、重命名或删除已有研究文件。

## 文件覆盖

`metadata/source_manifest.jsonl` 记录全部有效源文件的相对路径、字节数和 SHA-256。清单来自原始 `data/`、`Experiment/`、`KUSHULAN_dataset/`，不包含本次新增的说明、工具或上游代码。

源文件共 3,443 个，9,948,489,017 字节。其中 3,442 个文件直接在原路径跟踪；一个大型 ZIP 以四个分块保存。macOS 元数据、Office 锁文件不作为研究数据迁移。备份、优化器状态、随机状态及重复版本均保留。

## 大型 ZIP

原路径：`data/整理的数据-常用_副本/库淑兰剪纸数据采集/1.库淑兰剪纸作品.zip`。

大小：3,491,723,397 字节。SHA-256：

```text
a8f02489708248194324c1e71f620afeea501fb1572a715cc9b42500651c1830
```

GitHub Free 的 LFS 单文件上限为 2 GB，因此在 `archives/` 中保存三个 1 GiB 分块与一个尾块。分块不重新压缩、不解压文件；还原后与原始 ZIP 的字节完全一致。校验信息见 `archives/manifest.json`。

```bash
git lfs pull --include="*" --exclude=""
python tools/archive.py restore archives/manifest.json "data/整理的数据-常用_副本/库淑兰剪纸数据采集/1.库淑兰剪纸作品.zip"
python tools/inventory.py verify
```

还原工具在写入前检查每个分块，完成后再检查整体 SHA-256。若原路径已有正确文件，直接返回校验成功；若存在不同文件，则拒绝覆盖。

完整校验应输出 `Checked 3443 files; 0 failures`。缺少的图像、权重、分块或仍未下载的 LFS 指针都会被检出。

## 版本管理与验证

资料和权重采用 Git LFS；8 张较小的生成样例使用普通 Git，便于 README 直接展示。`.github/workflows/ci.yml` 在 Windows 与 Linux 上运行迁移工具测试，CI 不下载完整 LFS 资产，也不执行 GPU 训练。

```bash
python -m unittest discover -s tests -v
git lfs fsck
```

大文件传输中断后可以重新运行 `git push origin main`；LFS 通过内容哈希识别已经存在的对象。

GitHub 的 LFS 存储和下载流量按账号配额计算。克隆时建议先跳过大文件，再选择需要的目录。参考：[LFS 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[LFS 配额](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。
