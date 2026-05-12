# 1k 精标数据清洗与分析

本目录提供一个无第三方依赖的 Python 脚本，用于把服务器上的 1k 图片精标 CSV 整理为干净 JSONL，并输出一致性和异常类别分布统计。

## 服务器数据路径

默认路径来自 `服务器上数据信息.md`：

- 数据根目录：`/mnt/workspace/workgroup/leijian/benchmark/dataset/1k`
- manifest：`/mnt/workspace/workgroup/leijian/benchmark/dataset/1k/manifest.csv`
- 标注目录：`/mnt/workspace/workgroup/leijian/benchmark/dataset/1k/labels`
- 标注文件：`part1.csv` 到 `part5.csv`

## 推荐运行命令

在服务器上进入代码目录后运行：

```bash
python3 scripts/prepare_1k_annotations.py \
  --dataset-root /mnt/workspace/workgroup/leijian/benchmark/dataset/1k \
  --output-dir /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_columns \
  --source columns
```

默认说明：

- `--source columns` 使用 CSV 平铺列 `局部异常框选`、`有效性与整图异常选择` 作为主标注。
- 脚本仍会解析 `标注环节结果`，并在 `stats.md` 中报告两套来源是否冲突。
- 若确认 `标注环节结果` 才是复检后的主结果，可改跑：

```bash
python3 scripts/prepare_1k_annotations.py \
  --dataset-root /mnt/workspace/workgroup/leijian/benchmark/dataset/1k \
  --output-dir /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_mark_results \
  --source mark-results
```

## 输出文件

输出目录中会生成：

- `images.jsonl`：每张图一行，来自 `manifest.csv`，包含旧版三人粗标信息。
- `annotation_records.jsonl`：每条人工标注一行，约 2000 行，是主清洗结果。
- `local_boxes.jsonl`：局部异常 bbox 展开表，一框一行。
- `global_labels.jsonl`：整图异常展开表，一个整图异常标签一行。
- `image_summary.jsonl`：按图片聚合两人标注，包括投票、异常 code 计数、一致性辅助字段。
- `raw_rows.jsonl`：原始关键字段备份，便于追溯脏数据。
- `source_conflicts.jsonl`：平铺列与 `标注环节结果` 不一致的完整机器可读明细。
- `source_conflicts.md`：前若干条冲突的人类可读说明，默认最多 50 条，可用 `--max-conflict-examples` 调整。
- `stats.json`：机器可读统计。
- `stats.md`：人可读统计摘要，适合直接回传。

## 建议先回传的最小信息

服务器跑完后，优先把以下内容发回：

```bash
sed -n '1,220p' /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_columns/stats.md
```

如果 `source_conflicts` 很多，再补充：

```bash
grep -n "数据源冲突样例" -A 25 /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_columns/stats.md
```

如果脚本报错，回传完整终端输出，以及：

```bash
python3 --version
ls -lh /mnt/workspace/workgroup/leijian/benchmark/dataset/1k
ls -lh /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/labels
```

## 可视化少量 A/B 标注样例

先安装依赖：

```bash
pip install -r requirements.txt
```

默认从 `analysis_mark_results` 读取清洗后的 JSONL，按分层策略选择 20 张图：

```bash
python3 scripts/visualize_1k_annotations.py \
  --dataset-root /mnt/workspace/workgroup/leijian/benchmark/dataset/1k \
  --analysis-dir /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_mark_results \
  --output-dir /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/visualizations_mark_results_20 \
  --limit 20 \
  --selection balanced
```

每张选中图片会生成一个子目录，包含：

- `original.*`：原图。
- `bbox_A.jpg`：只画标注 A 的局部异常 bbox。
- `bbox_B.jpg`：只画标注 B 的局部异常 bbox。
- `bbox_A_B.jpg`：同时画 A/B 的 bbox；A 为实线，B 为虚线。
- `annotations.json`：A/B 的完整标注记录、局部/整图摘要、一致性指标。
- `annotations.md`：便于人工快速阅读的 A/B 标注摘要。

汇总文件：

- `selected_images.jsonl`
- `selected_images.md`

默认 `--selection balanced` 会优先覆盖：

- 有效性分歧样本。
- 平铺列与 `标注环节结果` 冲突样本。
- 局部异常高一致样本。
- 局部异常强分歧样本。
- 整图异常强分歧样本。
- 稀有或边界类别样本。
- 两人均未标异常的样本。

也可以指定模式：

```bash
--selection disagreement
--selection agreement
--selection random
```

或显式指定图片：

```bash
--image-ids 8,12,50
```

## 统计口径

- 有效性一致性：两条标注记录的 `图像有效性` 是否完全一致。
- 任意异常一致性：一条标注中只要存在任意局部异常或整图异常，即认为有异常。
- 局部异常类别统计：
  - `按 bbox 数`：同一标注里多个同类框会重复计数。
  - `按标注记录投票`：同一标注里同类多个框只计 1 票。
- 整图异常类别统计：
  - `按选择次数`：多选标签逐个计数。
  - `按标注记录投票`：同一标注里同一 code 只计 1 票。
- 多标签一致性：
  - `完全一致率`：两人选择的 code set 完全相同。
  - `平均 Jaccard`：两人 code set 的交并比平均值。
