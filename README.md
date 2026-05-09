# AIGI-QA Annotation Analysis

Utilities for cleaning and analyzing the 1k fine-grained annotation export for image-quality assessment of realistic AI-generated/person images.

The first workflow converts the raw annotation CSV files into normalized JSONL tables, then reports annotator consistency, abnormality distributions, parse warnings, and source conflicts.

## Quick Start

On the experiment server:

```bash
python3 scripts/prepare_1k_annotations.py \
  --dataset-root /mnt/workspace/workgroup/leijian/benchmark/dataset/1k \
  --output-dir /mnt/workspace/workgroup/leijian/benchmark/dataset/1k/analysis_columns \
  --source columns
```

Read the detailed workflow in [README_1k_annotation_analysis.md](README_1k_annotation_analysis.md).

## Notes

- The raw server data note `服务器上数据信息.md` is intentionally ignored and must not be committed.
- The script uses only the Python standard library.
- Label CSV files are expected to be GBK/cp936 encoded.
