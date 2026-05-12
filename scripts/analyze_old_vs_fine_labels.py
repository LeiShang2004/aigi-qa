#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze relationship between old coarse labels and new fine annotations.

The script reads server/path hints from ``服务器上数据信息.md`` when available,
then parses the original manifest + label CSVs twice:

- columns: ``局部异常框选`` + ``有效性与整图异常选择``
- mark-results: ``标注环节结果`` / MarkResult

Outputs are Markdown and JSON summaries for both sources.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_1k_annotations import (
    DEFAULT_DATASET_ROOT,
    GLOBAL_CODE_NAMES,
    LOCAL_CODE_NAMES,
    load_label_records,
    load_manifest,
    md_cell,
)


OLD_LABELS = [
    "可通过",
    "轻微物理逻辑错误",
    "明显物理逻辑错误",
    "轻微假感",
    "明显一眼假",
    "其他劣质AI",
    "无效数据",
]

OLD_LOGIC_LABELS = {"轻微物理逻辑错误", "明显物理逻辑错误"}
OLD_FAKE_LABELS = {"轻微假感", "明显一眼假"}
OLD_BAD_LABELS = OLD_LOGIC_LABELS | OLD_FAKE_LABELS | {"其他劣质AI", "无效数据"}
OLD_SEVERE_LABELS = {"明显物理逻辑错误", "明显一眼假", "其他劣质AI", "无效数据"}

FINE_LOGIC_LIKE_CODES = {
    "L01",
    "L02",
    "L04",
    "L05",
    "L06",
    "L07",
    "L08",
    "L10",
    "G02",
    "G05",
}

FINE_FAKE_LIKE_CODES = {
    "L03",
    "L09",
    "L11",
    "G01",
    "G03",
    "G04",
}


def read_server_info_dataset_root(path: Path) -> Path | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"`(/mnt/[^`]+?/dataset/1k)`",
        r"(/mnt/\S+?/dataset/1k)(?:/manifest\.csv|/labels|/images)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return Path(match.group(1))
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(value: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(value) * 100.0 / float(denominator), 4)


def old_majority_strict(old_labels: list[str]) -> str:
    counts = Counter(old_labels)
    if not counts:
        return "__missing__"
    label, count = counts.most_common(1)[0]
    return label if count >= 2 else "无多数"


def old_pattern(old_labels: list[str]) -> str:
    counts = Counter(old_labels)
    return " + ".join(f"{label}:{counts[label]}" for label in sorted(counts))


def old_features(old_labels: list[str]) -> dict[str, Any]:
    counts = Counter(old_labels)
    return {
        "old_labels": old_labels,
        "old_vote_counts": dict(counts),
        "old_majority_strict": old_majority_strict(old_labels),
        "old_pattern": old_pattern(old_labels),
        "old_unanimous": bool(counts and counts.most_common(1)[0][1] == len(old_labels)),
        "old_has_majority": bool(counts and counts.most_common(1)[0][1] >= 2),
        "old_bad_votes": sum(1 for label in old_labels if label in OLD_BAD_LABELS),
        "old_pass_votes": sum(1 for label in old_labels if label == "可通过"),
        "old_logic_votes": sum(1 for label in old_labels if label in OLD_LOGIC_LABELS),
        "old_fake_votes": sum(1 for label in old_labels if label in OLD_FAKE_LABELS),
        "old_severe_votes": sum(1 for label in old_labels if label in OLD_SEVERE_LABELS),
    }


def severity_weight(value: str | None) -> int:
    if value == "severe":
        return 2
    if value == "mild":
        return 1
    return 0


def code_set(record: dict[str, Any], field_name: str) -> set[str]:
    return {item["code"] for item in record.get(field_name, []) if item.get("code")}


def record_fine_score(record: dict[str, Any]) -> int:
    score = 0
    for item in record.get("local_abnormalities", []):
        score += 2 + severity_weight(item.get("severity"))
    for item in record.get("global_abnormalities", []):
        score += 1 + severity_weight(item.get("severity"))
    return score


def image_fine_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    local_bbox_counts: Counter = Counter()
    local_record_votes: Counter = Counter()
    global_selection_counts: Counter = Counter()
    global_record_votes: Counter = Counter()
    validity_votes: Counter = Counter()
    any_abnormal_votes = 0
    fine_score = 0
    source_conflict_records = 0

    for record in records:
        validity_votes[record.get("validity") or "__missing__"] += 1
        local_codes = code_set(record, "local_abnormalities")
        global_codes = code_set(record, "global_abnormalities")
        local_record_votes.update(local_codes)
        global_record_votes.update(global_codes)
        if local_codes or global_codes:
            any_abnormal_votes += 1
        fine_score += record_fine_score(record)
        if record.get("source_conflict"):
            source_conflict_records += 1

        for item in record.get("local_abnormalities", []):
            code = item.get("code")
            if code:
                local_bbox_counts[code] += 1
        for item in record.get("global_abnormalities", []):
            code = item.get("code")
            if code:
                global_selection_counts[code] += 1

    local_any_codes = set(local_record_votes)
    global_any_codes = set(global_record_votes)
    all_any_codes = local_any_codes | global_any_codes

    return {
        "num_records": len(records),
        "validity_votes": dict(validity_votes),
        "validity_majority": validity_votes.most_common(1)[0][0] if validity_votes else "__missing__",
        "validity_agree": len(validity_votes) == 1 if records else None,
        "any_abnormal_votes": any_abnormal_votes,
        "all_annotators_mark_abnormal": any_abnormal_votes == len(records) if records else False,
        "no_annotator_marks_abnormal": any_abnormal_votes == 0,
        "fine_score": fine_score,
        "local_bbox_total": sum(local_bbox_counts.values()),
        "global_label_total": sum(global_selection_counts.values()),
        "local_code_vote_counts": dict(sorted(local_record_votes.items())),
        "global_code_vote_counts": dict(sorted(global_record_votes.items())),
        "local_bbox_counts": dict(sorted(local_bbox_counts.items())),
        "global_selection_counts": dict(sorted(global_selection_counts.items())),
        "fine_logic_like_votes": sum(
            1
            for record in records
            if (code_set(record, "local_abnormalities") | code_set(record, "global_abnormalities"))
            & FINE_LOGIC_LIKE_CODES
        ),
        "fine_fake_like_votes": sum(
            1
            for record in records
            if (code_set(record, "local_abnormalities") | code_set(record, "global_abnormalities"))
            & FINE_FAKE_LIKE_CODES
        ),
        "has_L03_skin": int("L03" in local_any_codes),
        "has_L05_hand": int("L05" in local_any_codes),
        "has_G01_material": int("G01" in global_any_codes),
        "has_G04_style": int("G04" in global_any_codes),
        "source_conflict_records": source_conflict_records,
    }


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(rank_values(xs), rank_values(ys))


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def cross_tab(rows: list[dict[str, Any]], row_key: str, col_key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(row_key))][str(row.get(col_key))] += 1
    return {key: dict(counter) for key, counter in sorted(table.items())}


def counter_table(counter: Counter, denominator: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    total = sum(counter.values()) if denominator is None else denominator
    items = counter.most_common(limit)
    return [{"key": key, "count": count, "pct": pct(count, total)} for key, count in items]


def top_code_tables(rows: list[dict[str, Any]], old_group_key: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(old_group_key))].append(row)

    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group, group_rows in sorted(grouped.items()):
        local_counter: Counter = Counter()
        global_counter: Counter = Counter()
        for row in group_rows:
            local_counter.update(row.get("local_code_vote_counts") or {})
            global_counter.update(row.get("global_code_vote_counts") or {})
        denominator = max(1, len(group_rows) * 2)
        output[group] = {
            "local_top": counter_table(local_counter, denominator=denominator, limit=8),
            "global_top": counter_table(global_counter, denominator=denominator, limit=8),
        }
    return output


def grouped_numeric_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(row)

    summary = []
    for group, group_rows in sorted(grouped.items()):
        n = len(group_rows)
        summary.append(
            {
                "group": group,
                "n": n,
                "avg_old_bad_votes": round(statistics.mean(row["old_bad_votes"] for row in group_rows), 4),
                "avg_old_logic_votes": round(statistics.mean(row["old_logic_votes"] for row in group_rows), 4),
                "avg_old_fake_votes": round(statistics.mean(row["old_fake_votes"] for row in group_rows), 4),
                "avg_new_any_abnormal_votes": round(statistics.mean(row["any_abnormal_votes"] for row in group_rows), 4),
                "avg_fine_score": round(statistics.mean(row["fine_score"] for row in group_rows), 4),
                "avg_local_bbox_total": round(statistics.mean(row["local_bbox_total"] for row in group_rows), 4),
                "avg_global_label_total": round(statistics.mean(row["global_label_total"] for row in group_rows), 4),
                "avg_fine_logic_like_votes": round(statistics.mean(row["fine_logic_like_votes"] for row in group_rows), 4),
                "avg_fine_fake_like_votes": round(statistics.mean(row["fine_fake_like_votes"] for row in group_rows), 4),
            }
        )
    summary.sort(key=lambda item: (-item["n"], item["group"]))
    return summary


def correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("old_bad_votes", "any_abnormal_votes"),
        ("old_bad_votes", "fine_score"),
        ("old_bad_votes", "local_bbox_total"),
        ("old_bad_votes", "global_label_total"),
        ("old_logic_votes", "fine_logic_like_votes"),
        ("old_logic_votes", "local_bbox_total"),
        ("old_fake_votes", "fine_fake_like_votes"),
        ("old_fake_votes", "global_label_total"),
        ("old_fake_votes", "has_L03_skin"),
        ("old_fake_votes", "has_G01_material"),
        ("old_fake_votes", "has_G04_style"),
        ("old_logic_votes", "has_L05_hand"),
        ("old_severe_votes", "fine_score"),
    ]
    output = []
    for old_key, fine_key in pairs:
        xs = [float(row.get(old_key) or 0) for row in rows]
        ys = [float(row.get(fine_key) or 0) for row in rows]
        output.append(
            {
                "old_feature": old_key,
                "fine_feature": fine_key,
                "pearson": rounded(pearson(xs, ys)),
                "spearman": rounded(spearman(xs, ys)),
            }
        )
    return output


def build_image_rows(
    images: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {int(image["id"]): image for image in images if image.get("id") is not None}
    records_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_id = record.get("image_id")
        if image_id is not None:
            records_by_image[int(image_id)].append(record)

    rows: list[dict[str, Any]] = []
    for image_id in sorted(set(manifest_by_id) | set(records_by_image)):
        image = manifest_by_id.get(image_id, {})
        old = old_features(image.get("old_labels") or [])
        fine = image_fine_features(records_by_image.get(image_id, []))
        rows.append(
            {
                "image_id": image_id,
                "url": image.get("url"),
                "relative_path": image.get("relative_path"),
                **old,
                **fine,
            }
        )
    return rows


def analyze_source(
    source: str,
    images: list[dict[str, Any]],
    label_paths: list[Path],
    label_encodings: list[str],
) -> dict[str, Any]:
    records, _raw_rows, metadata = load_label_records(label_paths, label_encodings, source)
    rows = build_image_rows(images, records)

    old_majority_counter = Counter(row["old_majority_strict"] for row in rows)
    old_pattern_counter = Counter(row["old_pattern"] for row in rows)
    any_abnormal_counter = Counter(str(row["any_abnormal_votes"]) for row in rows)
    validity_counter = Counter(row["validity_majority"] for row in rows)
    source_conflict_counter = Counter(str(row["source_conflict_records"]) for row in rows)

    return {
        "source": source,
        "metadata": metadata,
        "counts": {
            "images": len(rows),
            "annotation_records": len(records),
            "images_with_two_records": sum(1 for row in rows if row["num_records"] == 2),
            "images_with_source_conflict": sum(1 for row in rows if row["source_conflict_records"] > 0),
        },
        "image_rows": rows,
        "old_majority_distribution": counter_table(old_majority_counter, denominator=len(rows)),
        "old_pattern_top": counter_table(old_pattern_counter, denominator=len(rows), limit=20),
        "new_any_abnormal_votes_distribution": counter_table(any_abnormal_counter, denominator=len(rows)),
        "new_validity_majority_distribution": counter_table(validity_counter, denominator=len(rows)),
        "source_conflict_records_per_image": counter_table(source_conflict_counter, denominator=len(rows)),
        "old_majority_by_new_any_abnormal_votes": cross_tab(rows, "old_majority_strict", "any_abnormal_votes"),
        "old_majority_by_new_validity_majority": cross_tab(rows, "old_majority_strict", "validity_majority"),
        "old_majority_group_numeric_summary": grouped_numeric_summary(rows, "old_majority_strict"),
        "top_fine_codes_by_old_majority": top_code_tables(rows, "old_majority_strict"),
        "correlations": correlation_summary(rows),
    }


def table_lines(title: str, rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["无数据", ""]
    if limit is not None:
        rows = rows[:limit]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_cell(row.get(column)) for column in columns) + " |")
    lines.append("")
    return lines


def render_cross_tab(title: str, table: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not table:
        return lines + ["无数据", ""]
    columns = sorted({column for row in table.values() for column in row})
    lines.append("| old_majority | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for row_key, row in table.items():
        lines.append("| " + md_cell(row_key) + " | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |")
    lines.append("")
    return lines


def render_top_codes_by_old_majority(title: str, top_codes: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    lines = [f"## {title}", ""]
    for group, value in top_codes.items():
        lines.append(f"### {group}")
        lines.append("")
        local = ", ".join(f"{item['key']}:{item['count']}({item['pct']}%)" for item in value.get("local_top", [])[:6]) or "无"
        global_ = ", ".join(f"{item['key']}:{item['count']}({item['pct']}%)" for item in value.get("global_top", [])[:6]) or "无"
        lines.append(f"- 局部 top：{local}")
        lines.append(f"- 整图 top：{global_}")
        lines.append("")
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 旧粗标与新精标关系分析",
        "",
        "分析对象：旧版 `标注1/标注2/标注3` 与新版局部/整图异常标注之间的关系。",
        "",
        "旧标签按图片聚合；新标签按图片聚合两位标注员的记录。",
        "",
        "注意：相关系数只描述这 1k 抽样中的共现关系，不等价于因果，也不代表类别定义天然一致。",
        "",
    ]

    for source_result in result["sources"]:
        source = source_result["source"]
        counts = source_result["counts"]
        lines.extend(
            [
                f"# Source: {source}",
                "",
                f"- 图片数：{counts['images']}",
                f"- 标注记录数：{counts['annotation_records']}",
                f"- 每图两条记录的图片数：{counts['images_with_two_records']}",
                f"- 含 columns/mark-results 冲突的图片数：{counts['images_with_source_conflict']}",
                "",
            ]
        )
        lines.extend(table_lines("旧标签严格多数分布", source_result["old_majority_distribution"], ["key", "count", "pct"]))
        lines.extend(table_lines("旧标签组合 Top", source_result["old_pattern_top"], ["key", "count", "pct"], limit=20))
        lines.extend(table_lines("新标注：每图有异常的标注员数量分布", source_result["new_any_abnormal_votes_distribution"], ["key", "count", "pct"]))
        lines.extend(table_lines("新标注：有效性多数分布", source_result["new_validity_majority_distribution"], ["key", "count", "pct"]))
        lines.extend(render_cross_tab("旧严格多数标签 x 新异常标注员数量", source_result["old_majority_by_new_any_abnormal_votes"]))
        lines.extend(render_cross_tab("旧严格多数标签 x 新有效性多数", source_result["old_majority_by_new_validity_majority"]))
        lines.extend(
            table_lines(
                "按旧严格多数标签分组的数值均值",
                source_result["old_majority_group_numeric_summary"],
                [
                    "group",
                    "n",
                    "avg_old_bad_votes",
                    "avg_old_logic_votes",
                    "avg_old_fake_votes",
                    "avg_new_any_abnormal_votes",
                    "avg_fine_score",
                    "avg_local_bbox_total",
                    "avg_global_label_total",
                    "avg_fine_logic_like_votes",
                    "avg_fine_fake_like_votes",
                ],
            )
        )
        lines.extend(
            table_lines(
                "旧标签票数与新精标特征相关系数",
                source_result["correlations"],
                ["old_feature", "fine_feature", "pearson", "spearman"],
            )
        )
        lines.extend(render_top_codes_by_old_majority("各旧严格多数标签下的新精标 Top code", source_result["top_fine_codes_by_old_majority"]))
    return "\n".join(lines)


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root)
    server_info_path = Path(args.server_info)
    parsed = read_server_info_dataset_root(server_info_path)
    if parsed is not None:
        return parsed
    return DEFAULT_DATASET_ROOT


def run(args: argparse.Namespace) -> int:
    dataset_root = resolve_dataset_root(args)
    manifest_path = Path(args.manifest) if args.manifest else dataset_root / "manifest.csv"
    labels_dir = Path(args.labels_dir) if args.labels_dir else dataset_root / "labels"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(labels_dir.glob("part*.csv")) or sorted(labels_dir.glob("*.csv"))
    if not label_paths:
        raise FileNotFoundError(f"no label csv files found under {labels_dir}")

    manifest_encodings = args.manifest_encodings.split(",")
    label_encodings = args.label_encodings.split(",")
    images, manifest_metadata = load_manifest(manifest_path, manifest_encodings)

    sources = []
    image_rows_by_source = {}
    for source in args.sources.split(","):
        source = source.strip()
        if not source:
            continue
        if source not in {"columns", "mark-results"}:
            raise ValueError(f"unsupported source: {source}")
        source_result = analyze_source(source, images, label_paths, label_encodings)
        sources.append({key: value for key, value in source_result.items() if key != "image_rows"})
        image_rows_by_source[source] = source_result["image_rows"]
        write_jsonl(output_dir / f"old_vs_fine_image_rows_{source}.jsonl", source_result["image_rows"])

    result = {
        "metadata": {
            "dataset_root": str(dataset_root),
            "server_info": args.server_info,
            "manifest": str(manifest_path),
            "labels_dir": str(labels_dir),
            "label_files": [str(path) for path in label_paths],
            **manifest_metadata,
        },
        "sources": sources,
    }
    write_json(output_dir / "old_vs_fine_stats.json", result)
    (output_dir / "old_vs_fine_stats.md").write_text(render_markdown(result), encoding="utf-8")

    print(f"dataset_root={dataset_root}")
    print(f"output_dir={output_dir}")
    print(f"summary={output_dir / 'old_vs_fine_stats.md'}")
    for source, rows in image_rows_by_source.items():
        print(f"{source}: images={len(rows)} rows={output_dir / f'old_vs_fine_image_rows_{source}.jsonl'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze old coarse labels vs new fine-grained annotations.")
    parser.add_argument("--server-info", default="服务器上数据信息.md", help="Markdown file containing server dataset paths.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root. If omitted, parsed from --server-info.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/old_vs_fine")
    parser.add_argument("--sources", default="columns,mark-results", help="Comma-separated sources: columns,mark-results")
    parser.add_argument("--manifest-encodings", default="utf-8-sig,utf-8,gbk,cp936")
    parser.add_argument("--label-encodings", default="gbk,cp936,utf-8-sig,utf-8")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
