#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean and analyze the 1k AIGI-QA annotation export.

This script intentionally uses only the Python standard library so it can run
on the weak-network experiment server without installing dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/mnt/workspace/workgroup/leijian/benchmark/dataset/1k")

LOCAL_CODE_NAMES = {
    "L01": "眼部异常",
    "L02": "面部其他异常",
    "L03": "皮肤质感异常",
    "L04": "头发 / 发际线异常",
    "L05": "手部异常",
    "L06": "身体 / 四肢 / 脚 / 比例异常",
    "L07": "衣物 / 配饰 / 道具异常",
    "L08": "关系异常（接触 / 穿插 / 支撑 / 遮挡）",
    "L09": "局部光影 / 反射 / 影子异常",
    "L10": "背景局部异常",
    "L11": "局部虚化 / 清晰度 / 边缘异常",
    "L99": "其他局部异常",
}

GLOBAL_CODE_NAMES = {
    "G01": "人像整体材质假",
    "G02": "人物与背景整体不融合",
    "G03": "全局光色 / 影调不协调",
    "G04": "全局景深 / 清晰度 / 成像风格异常",
    "G05": "整体空间 / 场景逻辑异常",
    "G99": "其他整图异常",
}

CODE_NAME_MAP = {**LOCAL_CODE_NAMES, **GLOBAL_CODE_NAMES}

VALID_SEVERITY = {
    "明显": "severe",
    "严重": "severe",
    "轻微/疑似": "mild",
    "轻微": "mild",
    "疑似": "mild",
}

JSON_EMPTY_VALUES = {"", "null", "none", "nan", "NaN", "NULL", "None"}


@dataclass
class ParsedAnnotation:
    validity: str | None = None
    globals: list[dict[str, Any]] = field(default_factory=list)
    locals: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_has_data: bool = False


def compact_key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+", "", text).strip()


def clean_header(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\ufeff", "").replace("\u200b", "").strip()


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def read_text_with_encodings(path: Path, encodings: list[str]) -> tuple[str, str]:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError(f"no encoding provided for {path}")


def sniff_csv_dialect(text: str) -> csv.Dialect:
    sample = text[: min(len(text), 32768)]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel()
        dialect.delimiter = "\t" if "\t" in sample and sample.count("\t") > sample.count(",") else ","
        return dialect


def read_csv_dicts(path: Path, encodings: list[str]) -> tuple[list[dict[str, str]], str, str]:
    text, encoding = read_text_with_encodings(path, encodings)
    dialect = sniff_csv_dialect(text)
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows: list[dict[str, str]] = []
    for raw_row in reader:
        row: dict[str, str] = {}
        for key, value in raw_row.items():
            cleaned_key = clean_header(key)
            if cleaned_key:
                row[cleaned_key] = "" if value is None else value
        if any(str(v).strip() for v in row.values()):
            rows.append(row)
    return rows, encoding, dialect.delimiter


def parse_json_cell(value: Any) -> tuple[Any | None, str | None]:
    text = normalize_text(value)
    if text is None or text in JSON_EMPTY_VALUES:
        return None, None
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}@{exc.pos}"


def get_label_value(labels: dict[str, Any], aliases: list[str]) -> Any:
    if not labels:
        return None
    for alias in aliases:
        if alias in labels:
            return labels[alias]
    compact_aliases = {compact_key(alias) for alias in aliases}
    for key, value in labels.items():
        if compact_key(key) in compact_aliases:
            return value
    return None


def normalize_code_label(value: Any, expected_prefix: str) -> tuple[str | None, str | None]:
    text = normalize_text(value)
    if text is None:
        return None, None
    match = re.search(r"([LG])\s*(\d{2})", text, flags=re.IGNORECASE)
    if not match:
        return None, text
    prefix = match.group(1).upper()
    number = match.group(2)
    code = f"{prefix}{number}"
    if expected_prefix and prefix != expected_prefix:
        return None, text
    tail = (text[: match.start()] + text[match.end() :]).strip()
    tail = re.sub(r"^[\s:：/|-]+", "", tail).strip()
    if not tail:
        tail = (LOCAL_CODE_NAMES if prefix == "L" else GLOBAL_CODE_NAMES).get(code)
    return code, tail


def normalize_severity(value: Any, expected_code_prefix: str | None = None) -> tuple[str | None, str | None, str | None]:
    raw = normalize_text(value)
    if raw is None:
        return None, None, "severity_missing"
    compact = compact_key(raw)
    for candidate, normalized in VALID_SEVERITY.items():
        if compact_key(candidate) == compact:
            return raw, normalized, None
    if expected_code_prefix and re.search(rf"{expected_code_prefix}\s*\d{{2}}", raw, flags=re.IGNORECASE):
        return raw, "unknown", "severity_looks_like_class_label"
    return raw, "unknown", "severity_unknown_value"


def annotation_object_kind(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    obj_type = normalize_text(obj.get("type"))
    tab_id = normalize_text(obj.get("tabId"))
    if obj_type == "ImageAnnotation":
        return "local"
    if obj_type == "CommonExtensions" or tab_id == "CommonExtensions":
        return "common"
    if "meta" in obj and "annotations" in obj:
        return "local"
    return None


def collect_other_text(labels: dict[str, Any]) -> dict[str, Any]:
    other: dict[str, Any] = {}
    for key, value in labels.items():
        key_text = str(key)
        if any(token in key_text for token in ("其他", "描述", "文本", "原因")):
            other[key_text] = value
    return other


def points_to_xyxy(points: Any) -> tuple[list[float] | None, str | None]:
    if not isinstance(points, list) or not points:
        return None, "bbox_points_missing"
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            xs.append(float(point["x"]))
            ys.append(float(point["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not xs or not ys:
        return None, "bbox_points_invalid"
    return [min(xs), min(ys), max(xs), max(ys)], None


def parse_common_extensions(obj: Any, source_name: str) -> ParsedAnnotation:
    parsed = ParsedAnnotation(source_has_data=True)
    if not isinstance(obj, dict):
        parsed.warnings.append(f"{source_name}:common_not_dict")
        return parsed
    annotations = obj.get("annotations") or []
    if not isinstance(annotations, list):
        parsed.warnings.append(f"{source_name}:common_annotations_not_list")
        annotations = []
    for annotation_index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            parsed.warnings.append(f"{source_name}:common_annotation_{annotation_index}_not_dict")
            continue
        labels = annotation.get("labels") or {}
        if not isinstance(labels, dict):
            parsed.warnings.append(f"{source_name}:common_annotation_{annotation_index}_labels_not_dict")
            labels = {}

        validity = normalize_text(get_label_value(labels, ["图像有效性"]))
        if validity:
            if parsed.validity and parsed.validity != validity:
                parsed.warnings.append(f"{source_name}:conflicting_validity:{parsed.validity}!={validity}")
            parsed.validity = validity

        severity_raw, severity_norm, severity_warning = normalize_severity(
            get_label_value(labels, ["严重度 / 明显度", "严重度/明显度"]), "G"
        )
        if severity_warning and get_label_value(labels, ["整图异常类型选择"]):
            parsed.warnings.append(f"{source_name}:global_{severity_warning}")

        global_values = get_label_value(labels, ["整图异常类型选择"])
        if global_values is None:
            global_list: list[Any] = []
        elif isinstance(global_values, list):
            global_list = global_values
        else:
            global_list = [global_values]
        for global_index, global_value in enumerate(global_list):
            code, label = normalize_code_label(global_value, "G")
            if not code:
                parsed.warnings.append(f"{source_name}:global_code_missing:{global_value}")
            parsed.globals.append(
                {
                    "source": source_name,
                    "annotation_index": annotation_index,
                    "global_index": global_index,
                    "code": code,
                    "label": label,
                    "raw_label": global_value,
                    "severity": severity_norm,
                    "severity_raw": severity_raw,
                    "other_text": collect_other_text(labels),
                }
            )
    return parsed


def parse_image_annotation(obj: Any, source_name: str) -> ParsedAnnotation:
    parsed = ParsedAnnotation(source_has_data=True)
    if not isinstance(obj, dict):
        parsed.warnings.append(f"{source_name}:image_annotation_not_dict")
        return parsed
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    width = meta.get("width")
    height = meta.get("height")
    annotations = obj.get("annotations") or []
    if not isinstance(annotations, list):
        parsed.warnings.append(f"{source_name}:local_annotations_not_list")
        annotations = []

    for local_index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            parsed.warnings.append(f"{source_name}:local_annotation_{local_index}_not_dict")
            continue
        labels = annotation.get("labels") or {}
        if not isinstance(labels, dict):
            labels = {}
            parsed.warnings.append(f"{source_name}:local_annotation_{local_index}_labels_not_dict")

        raw_type = get_label_value(labels, ["局部异常类型选择"]) or annotation.get("name")
        code, label = normalize_code_label(raw_type, "L")
        if not code:
            fallback_code, fallback_label = normalize_code_label(annotation.get("name"), "L")
            code = fallback_code
            label = label or fallback_label
        if not code:
            parsed.warnings.append(f"{source_name}:local_code_missing:{raw_type}")

        severity_raw, severity_norm, severity_warning = normalize_severity(
            get_label_value(labels, ["严重度/明显度", "严重度 / 明显度"]), "L"
        )
        if severity_warning:
            parsed.warnings.append(f"{source_name}:local_{local_index}_{severity_warning}")

        xyxy, bbox_warning = points_to_xyxy(annotation.get("points"))
        if bbox_warning:
            parsed.warnings.append(f"{source_name}:local_{local_index}_{bbox_warning}")

        parsed.locals.append(
            {
                "source": source_name,
                "local_index": local_index,
                "code": code,
                "label": label,
                "raw_label": raw_type,
                "severity": severity_norm,
                "severity_raw": severity_raw,
                "bbox_xyxy": xyxy,
                "points": annotation.get("points"),
                "image_width": width,
                "image_height": height,
                "annotation_id": annotation.get("id"),
                "annotation_name": annotation.get("name"),
                "other_text": collect_other_text(labels),
            }
        )
    return parsed


def merge_parsed(parts: list[ParsedAnnotation]) -> ParsedAnnotation:
    merged = ParsedAnnotation()
    for part in parts:
        if part.source_has_data:
            merged.source_has_data = True
        if part.validity:
            if merged.validity and merged.validity != part.validity:
                merged.warnings.append(f"merge:conflicting_validity:{merged.validity}!={part.validity}")
            merged.validity = part.validity
        merged.globals.extend(part.globals)
        merged.locals.extend(part.locals)
        merged.warnings.extend(part.warnings)
    return merged


def extract_from_columns(row: dict[str, str]) -> ParsedAnnotation:
    parts: list[ParsedAnnotation] = []
    warnings: list[str] = []
    candidates = [
        ("columns.local_cell", row.get("局部异常框选")),
        ("columns.common_cell", row.get("有效性与整图异常选择")),
    ]
    for source_name, cell in candidates:
        obj, error = parse_json_cell(cell)
        if error:
            if normalize_text(cell):
                warnings.append(f"{source_name}:{error}")
            continue
        kind = annotation_object_kind(obj)
        if kind == "local":
            parts.append(parse_image_annotation(obj, source_name))
        elif kind == "common":
            parts.append(parse_common_extensions(obj, source_name))
        elif obj is not None:
            warnings.append(f"{source_name}:unknown_annotation_object")
    parsed = merge_parsed(parts)
    parsed.warnings.extend(warnings)
    return parsed


def extract_from_mark_results(row: dict[str, str]) -> ParsedAnnotation:
    mark_results_obj, error = parse_json_cell(row.get("标注环节结果"))
    parts: list[ParsedAnnotation] = []
    warnings: list[str] = []
    if error:
        warnings.append(f"mark_results:{error}")
        parsed = ParsedAnnotation(warnings=warnings)
        return parsed
    if mark_results_obj is None:
        return ParsedAnnotation()
    if not isinstance(mark_results_obj, list):
        parsed = ParsedAnnotation(source_has_data=True)
        parsed.warnings.append("mark_results:not_list")
        return parsed
    for mark_index, mark_item in enumerate(mark_results_obj):
        if not isinstance(mark_item, dict):
            warnings.append(f"mark_results.item_{mark_index}:not_dict")
            continue
        mark_result_obj, mark_error = parse_json_cell(mark_item.get("MarkResult"))
        source_name = f"mark_results.item_{mark_index}"
        if mark_error:
            warnings.append(f"{source_name}:{mark_error}")
            continue
        kind = annotation_object_kind(mark_result_obj)
        if kind == "local":
            parts.append(parse_image_annotation(mark_result_obj, source_name))
        elif kind == "common":
            parts.append(parse_common_extensions(mark_result_obj, source_name))
        elif mark_result_obj is not None:
            warnings.append(f"{source_name}:unknown_annotation_object")
    parsed = merge_parsed(parts)
    parsed.warnings.extend(warnings)
    return parsed


def parsed_signature(parsed: ParsedAnnotation) -> dict[str, Any]:
    global_codes = sorted({item["code"] for item in parsed.globals if item.get("code")})
    local_codes = sorted({item["code"] for item in parsed.locals if item.get("code")})
    return {
        "validity": parsed.validity,
        "global_codes": global_codes,
        "local_codes": local_codes,
        "global_count": len(parsed.globals),
        "local_count": len(parsed.locals),
    }


def choose_parsed(
    columns: ParsedAnnotation,
    mark_results: ParsedAnnotation,
    source: str,
) -> tuple[str, ParsedAnnotation]:
    if source == "columns":
        if columns.source_has_data:
            return "columns", columns
        return "mark-results-fallback", mark_results
    if source == "mark-results":
        if mark_results.source_has_data:
            return "mark-results", mark_results
        return "columns-fallback", columns
    if source == "auto":
        if columns.source_has_data:
            return "columns", columns
        if mark_results.source_has_data:
            return "mark-results", mark_results
    return "none", ParsedAnnotation()


def as_int_or_none(value: Any) -> int | None:
    text = normalize_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_manifest_record(row: dict[str, str], row_index: int) -> dict[str, Any]:
    image_id = as_int_or_none(row.get("id"))
    old_labels = [normalize_text(row.get(name)) for name in ("标注1", "标注2", "标注3")]
    old_labels = [label for label in old_labels if label]
    old_counter = Counter(old_labels)
    old_majority = old_counter.most_common(1)[0][0] if old_counter else None
    old_unanimous = bool(old_counter and len(old_counter) == 1 and sum(old_counter.values()) >= 2)
    return {
        "id": image_id,
        "id_raw": row.get("id"),
        "manifest_row_index": row_index,
        "url": normalize_text(row.get("url")),
        "relative_path": normalize_text(row.get("relative_path")),
        "absolute_path": normalize_text(row.get("absolute_path")),
        "manifest_validity": normalize_text(row.get("validity")),
        "old_labels": old_labels,
        "old_label_counts": dict(old_counter),
        "old_majority_label": old_majority,
        "old_unanimous": old_unanimous,
    }


def normalize_label_record(
    row: dict[str, str],
    source_file: str,
    row_index: int,
    parse_source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_id = as_int_or_none(row.get("id"))
    columns = extract_from_columns(row)
    mark_results = extract_from_mark_results(row)
    chosen_source, chosen = choose_parsed(columns, mark_results, parse_source)
    columns_sig = parsed_signature(columns)
    mark_sig = parsed_signature(mark_results)
    source_conflict = bool(columns.source_has_data and mark_results.source_has_data and columns_sig != mark_sig)

    old_labels = [normalize_text(row.get(name)) for name in ("标注1", "标注2", "标注3")]
    old_labels = [label for label in old_labels if label]
    record_key = f"{source_file}:{row_index}"

    raw_subset = {
        "source_file": source_file,
        "row_index": row_index,
        "row_sha1": sha1_text(stable_json_dumps(row)),
        "id": row.get("id"),
        "url": row.get("url"),
        "标注1": row.get("标注1"),
        "标注2": row.get("标注2"),
        "标注3": row.get("标注3"),
        "样本ID": row.get("样本ID"),
        "局部异常框选": row.get("局部异常框选"),
        "有效性与整图异常选择": row.get("有效性与整图异常选择"),
        "标注环节结果": row.get("标注环节结果"),
        "标注环节人员": row.get("标注环节人员"),
    }

    record = {
        "record_key": record_key,
        "source_file": source_file,
        "row_index": row_index,
        "image_id": image_id,
        "image_id_raw": row.get("id"),
        "url": normalize_text(row.get("url")),
        "sample_id": normalize_text(row.get("样本ID")),
        "annotator": normalize_text(row.get("标注环节人员")),
        "old_labels": old_labels,
        "chosen_source": chosen_source,
        "validity": chosen.validity,
        "global_abnormalities": chosen.globals,
        "local_abnormalities": chosen.locals,
        "num_global_abnormalities": len(chosen.globals),
        "num_local_abnormalities": len(chosen.locals),
        "has_any_abnormality": bool(chosen.globals or chosen.locals),
        "parse_warnings": chosen.warnings,
        "columns_signature": columns_sig,
        "mark_results_signature": mark_sig,
        "source_conflict": source_conflict,
    }
    return record, raw_subset


def pct(value: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(value) * 100.0 / float(denominator), 4)


def counter_table(counter: Counter, denominator: int | None = None) -> list[dict[str, Any]]:
    total = sum(counter.values()) if denominator is None else denominator
    return [
        {"key": key, "count": count, "pct": pct(count, total)}
        for key, count in counter.most_common()
    ]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def cohen_kappa_bool(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    p_observed = agree / n
    p1_yes = sum(1 for a, _ in pairs if a) / n
    p2_yes = sum(1 for _, b in pairs if b) / n
    p_expected = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    if math.isclose(p_expected, 1.0):
        return 1.0 if math.isclose(p_observed, 1.0) else None
    return (p_observed - p_expected) / (1 - p_expected)


def code_set(record: dict[str, Any], field_name: str) -> set[str]:
    items = record.get(field_name) or []
    return {item["code"] for item in items if item.get("code")}


def make_image_summaries(
    records: list[dict[str, Any]],
    manifest_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_id = record.get("image_id")
        if image_id is not None:
            by_image[int(image_id)].append(record)

    summaries: list[dict[str, Any]] = []
    for image_id in sorted(set(manifest_by_id) | set(by_image)):
        image_records = sorted(by_image.get(image_id, []), key=lambda item: item["record_key"])
        validity_votes = Counter(record.get("validity") or "__missing__" for record in image_records)
        local_bbox_counts = Counter()
        local_record_votes = Counter()
        global_selection_counts = Counter()
        global_record_votes = Counter()

        for record in image_records:
            local_codes = code_set(record, "local_abnormalities")
            global_codes = code_set(record, "global_abnormalities")
            local_record_votes.update(local_codes)
            global_record_votes.update(global_codes)
            for item in record.get("local_abnormalities") or []:
                if item.get("code"):
                    local_bbox_counts[item["code"]] += 1
            for item in record.get("global_abnormalities") or []:
                if item.get("code"):
                    global_selection_counts[item["code"]] += 1

        manifest = manifest_by_id.get(image_id, {})
        summaries.append(
            {
                "image_id": image_id,
                "url": manifest.get("url") or (image_records[0].get("url") if image_records else None),
                "relative_path": manifest.get("relative_path"),
                "absolute_path": manifest.get("absolute_path"),
                "num_annotation_records": len(image_records),
                "record_keys": [record["record_key"] for record in image_records],
                "annotators": [record.get("annotator") for record in image_records],
                "validity_votes": dict(validity_votes),
                "validity_agree": len(validity_votes) == 1 if len(image_records) >= 2 else None,
                "has_any_abnormality_votes": Counter(str(bool(record.get("has_any_abnormality"))) for record in image_records),
                "local_bbox_counts": dict(local_bbox_counts),
                "local_record_votes": dict(local_record_votes),
                "global_selection_counts": dict(global_selection_counts),
                "global_record_votes": dict(global_record_votes),
                "old_labels": manifest.get("old_labels"),
                "old_majority_label": manifest.get("old_majority_label"),
                "old_unanimous": manifest.get("old_unanimous"),
            }
        )
    return summaries


def pairwise_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_id = record.get("image_id")
        if image_id is not None:
            by_image[int(image_id)].append(record)

    exact_two_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    more_than_two = 0
    for image_records in by_image.values():
        image_records = sorted(image_records, key=lambda item: item["record_key"])
        if len(image_records) == 2:
            exact_two_pairs.append((image_records[0], image_records[1]))
        elif len(image_records) > 2:
            more_than_two += 1

    validity_pairs = [
        (left.get("validity"), right.get("validity"))
        for left, right in exact_two_pairs
        if left.get("validity") is not None and right.get("validity") is not None
    ]
    binary_pairs = [
        (bool(left.get("has_any_abnormality")), bool(right.get("has_any_abnormality")))
        for left, right in exact_two_pairs
    ]

    local_jaccards: list[float] = []
    global_jaccards: list[float] = []
    local_exact = 0
    global_exact = 0
    for left, right in exact_two_pairs:
        left_local = code_set(left, "local_abnormalities")
        right_local = code_set(right, "local_abnormalities")
        left_global = code_set(left, "global_abnormalities")
        right_global = code_set(right, "global_abnormalities")
        local_jaccards.append(jaccard(left_local, right_local))
        global_jaccards.append(jaccard(left_global, right_global))
        local_exact += int(left_local == right_local)
        global_exact += int(left_global == right_global)

    def per_code_stats(field_name: str) -> list[dict[str, Any]]:
        observed_codes: set[str] = set()
        for left, right in exact_two_pairs:
            observed_codes.update(code_set(left, field_name))
            observed_codes.update(code_set(right, field_name))
        rows: list[dict[str, Any]] = []
        for code in sorted(observed_codes):
            pairs = [
                (code in code_set(left, field_name), code in code_set(right, field_name))
                for left, right in exact_two_pairs
            ]
            both_yes = sum(1 for a, b in pairs if a and b)
            left_only = sum(1 for a, b in pairs if a and not b)
            right_only = sum(1 for a, b in pairs if b and not a)
            both_no = sum(1 for a, b in pairs if not a and not b)
            rows.append(
                {
                    "code": code,
                    "name": CODE_NAME_MAP.get(code),
                    "n": len(pairs),
                    "both_yes": both_yes,
                    "left_only": left_only,
                    "right_only": right_only,
                    "both_no": both_no,
                    "agree_rate": pct(both_yes + both_no, len(pairs)),
                    "positive_agree_rate_among_any_positive": pct(both_yes, both_yes + left_only + right_only),
                    "kappa": None if cohen_kappa_bool(pairs) is None else round(cohen_kappa_bool(pairs), 6),
                }
            )
        rows.sort(key=lambda item: (item["both_yes"] + item["left_only"] + item["right_only"], item["code"]), reverse=True)
        return rows

    return {
        "images_with_records": len(by_image),
        "images_with_exactly_two_records": len(exact_two_pairs),
        "images_with_more_than_two_records": more_than_two,
        "validity_comparable_pairs": len(validity_pairs),
        "validity_exact_agree": sum(1 for a, b in validity_pairs if a == b),
        "validity_exact_agree_rate": pct(sum(1 for a, b in validity_pairs if a == b), len(validity_pairs)),
        "binary_abnormal_agree_rate": pct(sum(1 for a, b in binary_pairs if a == b), len(binary_pairs)),
        "binary_abnormal_kappa": None if cohen_kappa_bool(binary_pairs) is None else round(cohen_kappa_bool(binary_pairs), 6),
        "local_code_set_exact_agree_rate": pct(local_exact, len(exact_two_pairs)),
        "local_code_set_mean_jaccard": round(statistics.mean(local_jaccards), 6) if local_jaccards else None,
        "global_code_set_exact_agree_rate": pct(global_exact, len(exact_two_pairs)),
        "global_code_set_mean_jaccard": round(statistics.mean(global_jaccards), 6) if global_jaccards else None,
        "local_per_code": per_code_stats("local_abnormalities"),
        "global_per_code": per_code_stats("global_abnormalities"),
    }


def build_stats(
    images: list[dict[str, Any]],
    records: list[dict[str, Any]],
    image_summaries: list[dict[str, Any]],
    source_conflict_examples: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    validity_counter = Counter(record.get("validity") or "__missing__" for record in records)
    old_label_counter = Counter(label for image in images for label in (image.get("old_labels") or []))
    annotator_counter = Counter(record.get("annotator") or "__missing__" for record in records)
    chosen_source_counter = Counter(record.get("chosen_source") for record in records)
    warning_counter = Counter(warning for record in records for warning in (record.get("parse_warnings") or []))

    local_bbox_counter = Counter()
    local_record_vote_counter = Counter()
    local_severity_counter = Counter()
    global_selection_counter = Counter()
    global_record_vote_counter = Counter()
    global_severity_counter = Counter()

    for record in records:
        local_codes = set()
        global_codes = set()
        for item in record.get("local_abnormalities") or []:
            code = item.get("code") or "__missing__"
            local_bbox_counter[code] += 1
            local_codes.add(code)
            local_severity_counter[f"{code}|{item.get('severity') or '__missing__'}"] += 1
        for item in record.get("global_abnormalities") or []:
            code = item.get("code") or "__missing__"
            global_selection_counter[code] += 1
            global_codes.add(code)
            global_severity_counter[f"{code}|{item.get('severity') or '__missing__'}"] += 1
        local_record_vote_counter.update(local_codes)
        global_record_vote_counter.update(global_codes)

    records_per_image_counter = Counter(summary["num_annotation_records"] for summary in image_summaries)
    source_conflicts = sum(1 for record in records if record.get("source_conflict"))

    consistency = pairwise_consistency(records)

    return {
        "metadata": metadata,
        "counts": {
            "manifest_images": len(images),
            "annotation_records": len(records),
            "image_summaries": len(image_summaries),
            "source_conflicts": source_conflicts,
            "records_with_parse_warnings": sum(1 for record in records if record.get("parse_warnings")),
        },
        "records_per_image": counter_table(records_per_image_counter, len(image_summaries)),
        "chosen_source_distribution": counter_table(chosen_source_counter, len(records)),
        "annotator_distribution": counter_table(annotator_counter, len(records)),
        "manifest_old_label_distribution": counter_table(old_label_counter),
        "record_validity_distribution": counter_table(validity_counter, len(records)),
        "local_abnormalities_by_bbox_count": counter_table(local_bbox_counter),
        "local_abnormalities_by_record_vote": counter_table(local_record_vote_counter, len(records)),
        "global_abnormalities_by_selection_count": counter_table(global_selection_counter),
        "global_abnormalities_by_record_vote": counter_table(global_record_vote_counter, len(records)),
        "local_severity_distribution": counter_table(local_severity_counter),
        "global_severity_distribution": counter_table(global_severity_counter),
        "top_parse_warnings": counter_table(warning_counter),
        "consistency": consistency,
        "source_conflict_examples": source_conflict_examples,
    }


def table_lines(title: str, rows: list[dict[str, Any]], limit: int = 20) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["无数据", ""]
    lines.extend(["| key | count | pct |", "| --- | ---: | ---: |"])
    for row in rows[:limit]:
        lines.append(f"| {row.get('key')} | {row.get('count')} | {row.get('pct')} |")
    lines.append("")
    return lines


def code_table_lines(title: str, rows: list[dict[str, Any]], limit: int = 20) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["无数据", ""]
    lines.extend(["| code | name | count | pct |", "| --- | --- | ---: | ---: |"])
    for row in rows[:limit]:
        key = row.get("key")
        name = CODE_NAME_MAP.get(key, "")
        lines.append(f"| {key} | {name} | {row.get('count')} | {row.get('pct')} |")
    lines.append("")
    return lines


def per_code_consistency_lines(title: str, rows: list[dict[str, Any]], limit: int = 20) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["无数据", ""]
    lines.extend(
        [
            "| code | name | any_positive | both_yes | left_only | right_only | agree_rate | positive_agree_rate | kappa |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows[:limit]:
        any_positive = row["both_yes"] + row["left_only"] + row["right_only"]
        lines.append(
            f"| {row['code']} | {row.get('name') or ''} | {any_positive} | {row['both_yes']} | "
            f"{row['left_only']} | {row['right_only']} | {row['agree_rate']} | "
            f"{row['positive_agree_rate_among_any_positive']} | {row['kappa']} |"
        )
    lines.append("")
    return lines


def render_stats_markdown(stats: dict[str, Any]) -> str:
    counts = stats["counts"]
    consistency = stats["consistency"]
    lines = [
        "# 1k 精标清洗与分析摘要",
        "",
        "## 基本数量",
        "",
        f"- manifest 图片数：{counts['manifest_images']}",
        f"- 标注记录数：{counts['annotation_records']}",
        f"- 聚合图片数：{counts['image_summaries']}",
        f"- 平铺列与标注环节结果存在差异的记录数：{counts['source_conflicts']}",
        f"- 有解析 warning 的记录数：{counts['records_with_parse_warnings']}",
        "",
        "## 两人一致性",
        "",
        f"- 恰好两条标注记录的图片数：{consistency['images_with_exactly_two_records']}",
        f"- 有效性完全一致率：{consistency['validity_exact_agree_rate']}%",
        f"- 是否存在任意异常的一致率：{consistency['binary_abnormal_agree_rate']}%",
        f"- 是否存在任意异常的 Cohen kappa：{consistency['binary_abnormal_kappa']}",
        f"- 局部异常 code set 完全一致率：{consistency['local_code_set_exact_agree_rate']}%",
        f"- 局部异常 code set 平均 Jaccard：{consistency['local_code_set_mean_jaccard']}",
        f"- 整图异常 code set 完全一致率：{consistency['global_code_set_exact_agree_rate']}%",
        f"- 整图异常 code set 平均 Jaccard：{consistency['global_code_set_mean_jaccard']}",
        "",
    ]
    lines.extend(table_lines("每图标注记录数分布", stats["records_per_image"]))
    lines.extend(table_lines("采用数据源分布", stats["chosen_source_distribution"]))
    lines.extend(table_lines("标注人员分布", stats["annotator_distribution"]))
    lines.extend(table_lines("旧版三人粗标标签分布", stats["manifest_old_label_distribution"]))
    lines.extend(table_lines("有效性分布（按标注记录）", stats["record_validity_distribution"]))
    lines.extend(code_table_lines("局部异常最多类别（按 bbox 数）", stats["local_abnormalities_by_bbox_count"]))
    lines.extend(code_table_lines("局部异常最多类别（按标注记录投票）", stats["local_abnormalities_by_record_vote"]))
    lines.extend(code_table_lines("整图异常最多类别（按选择次数）", stats["global_abnormalities_by_selection_count"]))
    lines.extend(code_table_lines("整图异常最多类别（按标注记录投票）", stats["global_abnormalities_by_record_vote"]))
    lines.extend(table_lines("局部异常严重度分布（code|severity）", stats["local_severity_distribution"]))
    lines.extend(table_lines("整图异常严重度分布（code|severity）", stats["global_severity_distribution"]))
    lines.extend(per_code_consistency_lines("局部异常逐类别一致性", consistency["local_per_code"]))
    lines.extend(per_code_consistency_lines("整图异常逐类别一致性", consistency["global_per_code"]))
    lines.extend(table_lines("解析 warning Top", stats["top_parse_warnings"], limit=30))
    if stats["source_conflict_examples"]:
        lines.extend(["## 数据源冲突样例", ""])
        lines.extend(["| record_key | image_id | columns | mark_results |", "| --- | ---: | --- | --- |"])
        for item in stats["source_conflict_examples"][:20]:
            lines.append(
                f"| {item['record_key']} | {item['image_id']} | "
                f"`{stable_json_dumps(item['columns_signature'])}` | "
                f"`{stable_json_dumps(item['mark_results_signature'])}` |"
            )
        lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(manifest_path: Path, manifest_encodings: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, encoding, delimiter = read_csv_dicts(manifest_path, manifest_encodings)
    images = [normalize_manifest_record(row, index + 1) for index, row in enumerate(rows)]
    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_encoding": encoding,
        "manifest_delimiter": delimiter,
    }
    return images, metadata


def load_label_records(
    label_paths: list[Path],
    label_encodings: list[str],
    parse_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    file_metadata: list[dict[str, Any]] = []
    for label_path in label_paths:
        rows, encoding, delimiter = read_csv_dicts(label_path, label_encodings)
        file_metadata.append(
            {
                "path": str(label_path),
                "encoding": encoding,
                "delimiter": delimiter,
                "rows": len(rows),
            }
        )
        for index, row in enumerate(rows, 1):
            record, raw_subset = normalize_label_record(row, label_path.name, index, parse_source)
            all_records.append(record)
            raw_rows.append(raw_subset)
    return all_records, raw_rows, {"label_files": file_metadata}


def expanded_local_boxes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for box_index, item in enumerate(record.get("local_abnormalities") or []):
            rows.append(
                {
                    "record_key": record["record_key"],
                    "image_id": record.get("image_id"),
                    "url": record.get("url"),
                    "annotator": record.get("annotator"),
                    "box_index": box_index,
                    **item,
                }
            )
    return rows


def expanded_global_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for global_index, item in enumerate(record.get("global_abnormalities") or []):
            rows.append(
                {
                    "record_key": record["record_key"],
                    "image_id": record.get("image_id"),
                    "url": record.get("url"),
                    "annotator": record.get("annotator"),
                    "global_index": global_index,
                    **item,
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest) if args.manifest else dataset_root / "manifest.csv"
    labels_dir = Path(args.labels_dir) if args.labels_dir else dataset_root / "labels"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.label_files:
        label_paths = [Path(path) for path in args.label_files]
    else:
        label_paths = sorted(labels_dir.glob("part*.csv"))
        if not label_paths:
            label_paths = sorted(labels_dir.glob("*.csv"))

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    missing_labels = [path for path in label_paths if not path.exists()]
    if missing_labels:
        raise FileNotFoundError(f"label files not found: {missing_labels}")
    if not label_paths:
        raise FileNotFoundError(f"no label csv files found under: {labels_dir}")

    manifest_encodings = args.manifest_encodings.split(",")
    label_encodings = args.label_encodings.split(",")

    images, manifest_metadata = load_manifest(manifest_path, manifest_encodings)
    records, raw_rows, label_metadata = load_label_records(label_paths, label_encodings, args.source)
    manifest_by_id = {image["id"]: image for image in images if image.get("id") is not None}
    image_summaries = make_image_summaries(records, manifest_by_id)
    local_boxes = expanded_local_boxes(records)
    global_labels = expanded_global_labels(records)

    source_conflict_examples = [
        {
            "record_key": record["record_key"],
            "image_id": record.get("image_id"),
            "columns_signature": record.get("columns_signature"),
            "mark_results_signature": record.get("mark_results_signature"),
        }
        for record in records
        if record.get("source_conflict")
    ][: args.max_conflict_examples]

    metadata = {
        "parse_source": args.source,
        **manifest_metadata,
        **label_metadata,
        "outputs": {
            "images": "images.jsonl",
            "annotation_records": "annotation_records.jsonl",
            "local_boxes": "local_boxes.jsonl",
            "global_labels": "global_labels.jsonl",
            "image_summary": "image_summary.jsonl",
            "raw_rows": "raw_rows.jsonl",
            "stats_json": "stats.json",
            "stats_markdown": "stats.md",
        },
    }
    stats = build_stats(images, records, image_summaries, source_conflict_examples, metadata)

    write_jsonl(output_dir / "images.jsonl", images)
    write_jsonl(output_dir / "annotation_records.jsonl", records)
    write_jsonl(output_dir / "local_boxes.jsonl", local_boxes)
    write_jsonl(output_dir / "global_labels.jsonl", global_labels)
    write_jsonl(output_dir / "image_summary.jsonl", image_summaries)
    write_jsonl(output_dir / "raw_rows.jsonl", raw_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "stats.md").write_text(render_stats_markdown(stats), encoding="utf-8")

    print(f"wrote: {output_dir}")
    print(f"manifest_images={len(images)} annotation_records={len(records)}")
    print(f"local_boxes={len(local_boxes)} global_labels={len(global_labels)}")
    print(f"source_conflicts={stats['counts']['source_conflicts']} records_with_warnings={stats['counts']['records_with_parse_warnings']}")
    print(f"summary: {output_dir / 'stats.md'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and analyze the 1k AIGI-QA annotation CSV export.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root containing manifest.csv and labels/.")
    parser.add_argument("--manifest", default=None, help="Override manifest.csv path.")
    parser.add_argument("--labels-dir", default=None, help="Override labels directory.")
    parser.add_argument("--label-files", nargs="*", default=None, help="Explicit label CSV paths. Defaults to labels/part*.csv.")
    parser.add_argument("--output-dir", default="outputs/1k_annotation_analysis", help="Directory for cleaned JSONL and stats outputs.")
    parser.add_argument(
        "--source",
        choices=["columns", "mark-results", "auto"],
        default="columns",
        help="Canonical source. columns means the flattened CSV columns; mark-results means 标注环节结果; auto prefers columns then falls back.",
    )
    parser.add_argument(
        "--manifest-encodings",
        default="utf-8-sig,utf-8,gbk,cp936",
        help="Comma-separated encoding fallbacks for manifest.csv.",
    )
    parser.add_argument(
        "--label-encodings",
        default="gbk,cp936,utf-8-sig,utf-8",
        help="Comma-separated encoding fallbacks for label CSV files.",
    )
    parser.add_argument("--max-conflict-examples", type=int, default=50)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
