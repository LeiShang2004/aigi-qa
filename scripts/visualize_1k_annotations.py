#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select and visualize a small set of 1k annotation examples.

Inputs are the JSONL files produced by prepare_1k_annotations.py. For each
selected image this script writes four visual versions:

1. original image
2. annotator A local bboxes
3. annotator B local bboxes
4. annotator A+B local bboxes

It also writes per-image A/B annotation summaries for quick qualitative review.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - exercised on servers without Pillow
    print("ERROR: Pillow is required. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


DEFAULT_DATASET_ROOT = Path("/mnt/workspace/workgroup/leijian/benchmark/dataset/1k")
DEFAULT_ANALYSIS_SUBDIR = "analysis_mark_results"

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

CODE_COLORS = {
    "L01": (239, 139, 180),
    "L02": (246, 189, 22),
    "L03": (247, 89, 171),
    "L04": (255, 120, 117),
    "L05": (255, 153, 102),
    "L06": (255, 169, 64),
    "L07": (255, 158, 123),
    "L08": (22, 119, 255),
    "L09": (231, 76, 60),
    "L10": (155, 89, 182),
    "L11": (243, 156, 18),
    "L99": (96, 96, 96),
}

RARE_LOCAL_CODES = {"L08", "L09", "L99", "L10", "L11"}
RARE_GLOBAL_CODES = {"G05", "G99", "G02"}
COMMON_PRIORITY = ["L05", "L03", "L01", "L02", "G01", "G04", "G03"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl: {exc}") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def code_set(record: dict[str, Any], field_name: str) -> set[str]:
    return {item["code"] for item in record.get(field_name, []) if item.get("code")}


def code_counts(record: dict[str, Any], field_name: str) -> Counter:
    counter: Counter = Counter()
    for item in record.get(field_name, []):
        code = item.get("code")
        if code:
            counter[code] += 1
    return counter


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def severity_score(value: str | None) -> int:
    if value == "severe":
        return 2
    if value == "mild":
        return 1
    return 0


def abnormality_score(record: dict[str, Any]) -> int:
    score = 0
    for item in record.get("local_abnormalities", []):
        score += 2 + severity_score(item.get("severity"))
    for item in record.get("global_abnormalities", []):
        score += 1 + severity_score(item.get("severity"))
    return score


def load_inputs(analysis_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images_path = analysis_dir / "images.jsonl"
    records_path = analysis_dir / "annotation_records.jsonl"
    if not images_path.exists():
        raise FileNotFoundError(f"missing {images_path}; run prepare_1k_annotations.py first")
    if not records_path.exists():
        raise FileNotFoundError(f"missing {records_path}; run prepare_1k_annotations.py first")

    images = {}
    for image in read_jsonl(images_path):
        image_id = image.get("id")
        if image_id is not None:
            images[int(image_id)] = image

    records_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(records_path):
        image_id = record.get("image_id")
        if image_id is not None:
            records_by_image[int(image_id)].append(record)

    for image_id in list(records_by_image):
        records_by_image[image_id].sort(key=lambda item: item.get("record_key") or "")
    return images, records_by_image


def image_path_for(image: dict[str, Any], dataset_root: Path) -> Path | None:
    absolute = image.get("absolute_path")
    if absolute:
        path = Path(str(absolute))
        if path.exists():
            return path
    relative = image.get("relative_path")
    if relative:
        path = dataset_root / str(relative)
        if path.exists():
            return path
    return None


def build_candidates(
    images: dict[int, dict[str, Any]],
    records_by_image: dict[int, list[dict[str, Any]]],
    dataset_root: Path,
    include_missing_images: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for image_id, records in records_by_image.items():
        if len(records) < 2:
            continue
        image = images.get(image_id, {})
        path = image_path_for(image, dataset_root)
        if path is None and not include_missing_images:
            continue
        a_record, b_record = records[0], records[1]
        a_local = code_set(a_record, "local_abnormalities")
        b_local = code_set(b_record, "local_abnormalities")
        a_global = code_set(a_record, "global_abnormalities")
        b_global = code_set(b_record, "global_abnormalities")
        local_union = a_local | b_local
        global_union = a_global | b_global
        local_intersection = a_local & b_local
        global_intersection = a_global & b_global
        total_local_boxes = len(a_record.get("local_abnormalities", [])) + len(b_record.get("local_abnormalities", []))
        total_global_labels = len(a_record.get("global_abnormalities", [])) + len(b_record.get("global_abnormalities", []))
        total_score = abnormality_score(a_record) + abnormality_score(b_record)
        validity_a = a_record.get("validity")
        validity_b = b_record.get("validity")
        candidate = {
            "image_id": image_id,
            "image": image,
            "image_path": str(path) if path else None,
            "records": [a_record, b_record],
            "a_record_key": a_record.get("record_key"),
            "b_record_key": b_record.get("record_key"),
            "a_annotator": a_record.get("annotator"),
            "b_annotator": b_record.get("annotator"),
            "validity_a": validity_a,
            "validity_b": validity_b,
            "validity_agree": validity_a == validity_b,
            "a_local_codes": sorted(a_local),
            "b_local_codes": sorted(b_local),
            "a_global_codes": sorted(a_global),
            "b_global_codes": sorted(b_global),
            "local_union": sorted(local_union),
            "global_union": sorted(global_union),
            "local_intersection": sorted(local_intersection),
            "global_intersection": sorted(global_intersection),
            "local_jaccard": round(jaccard(a_local, b_local), 6),
            "global_jaccard": round(jaccard(a_global, b_global), 6),
            "total_local_boxes": total_local_boxes,
            "total_global_labels": total_global_labels,
            "total_abnormality_score": total_score,
            "has_no_abnormality": total_local_boxes == 0 and total_global_labels == 0,
            "has_source_conflict": bool(a_record.get("source_conflict") or b_record.get("source_conflict")),
            "has_rare_local": bool(local_union & RARE_LOCAL_CODES),
            "has_rare_global": bool(global_union & RARE_GLOBAL_CODES),
            "has_common_priority": bool((local_union | global_union) & set(COMMON_PRIORITY)),
            "old_majority_label": image.get("old_majority_label"),
            "old_unanimous": image.get("old_unanimous"),
        }
        candidates.append(candidate)
    return candidates


def add_from_pool(
    selected: list[dict[str, Any]],
    used_ids: set[int],
    pool: list[dict[str, Any]],
    quota: int,
    reason: str,
) -> None:
    for candidate in pool:
        if len(selected) >= quota:
            break
        image_id = candidate["image_id"]
        if image_id in used_ids:
            continue
        selected.append({**candidate, "selection_reason": reason})
        used_ids.add(image_id)


def select_balanced(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    strata = [
        (
            "validity_disagreement",
            2,
            [
                item
                for item in candidates
                if not item["validity_agree"]
            ],
            lambda item: (-item["total_abnormality_score"], item["local_jaccard"], item["image_id"]),
        ),
        (
            "high_local_agreement",
            3,
            [
                item
                for item in candidates
                if item["local_intersection"] and item["local_jaccard"] >= 0.5
            ],
            lambda item: (
                -len(item["local_intersection"]),
                -item["total_local_boxes"],
                -item["total_abnormality_score"],
                item["image_id"],
            ),
        ),
        (
            "source_conflict",
            3,
            [
                item
                for item in candidates
                if item["has_source_conflict"]
            ],
            lambda item: (
                item["local_jaccard"],
                item["global_jaccard"],
                -item["total_abnormality_score"],
                item["image_id"],
            ),
        ),
        (
            "high_local_disagreement",
            4,
            [
                item
                for item in candidates
                if item["local_union"] and item["local_jaccard"] <= 0.2
            ],
            lambda item: (
                item["local_jaccard"],
                -len(item["local_union"]),
                -item["total_local_boxes"],
                item["image_id"],
            ),
        ),
        (
            "high_global_disagreement",
            3,
            [
                item
                for item in candidates
                if item["global_union"] and item["global_jaccard"] <= 0.25
            ],
            lambda item: (
                item["global_jaccard"],
                -len(item["global_union"]),
                -item["total_global_labels"],
                item["image_id"],
            ),
        ),
        (
            "rare_or_edge_category",
            3,
            [
                item
                for item in candidates
                if item["has_rare_local"] or item["has_rare_global"]
            ],
            lambda item: (
                -int(item["has_rare_local"]),
                -int(item["has_rare_global"]),
                item["local_jaccard"],
                item["global_jaccard"],
                item["image_id"],
            ),
        ),
        (
            "no_abnormality_agreement",
            2,
            [
                item
                for item in candidates
                if item["has_no_abnormality"] and item["validity_agree"]
            ],
            lambda item: (
                0 if item.get("old_majority_label") == "可通过" else 1,
                item["image_id"],
            ),
        ),
    ]

    target = 0
    for reason, quota, pool, key_fn in strata:
        if len(selected) >= limit:
            break
        target = min(limit, target + quota)
        add_from_pool(selected, used_ids, sorted(pool, key=key_fn), target, reason)

    fill_pool = sorted(
        candidates,
        key=lambda item: (
            -item["total_abnormality_score"],
            item["local_jaccard"],
            item["global_jaccard"],
            item["image_id"],
        ),
    )
    add_from_pool(selected, used_ids, fill_pool, limit, "score_fill")
    return selected[:limit]


def select_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
    mode: str,
    seed: int,
    explicit_ids: list[int] | None,
) -> list[dict[str, Any]]:
    if explicit_ids:
        by_id = {item["image_id"]: item for item in candidates}
        selected = []
        for image_id in explicit_ids[:limit]:
            if image_id in by_id:
                selected.append({**by_id[image_id], "selection_reason": "explicit_id"})
        return selected

    if mode == "balanced":
        return select_balanced(candidates, limit)
    if mode == "disagreement":
        return [
            {**item, "selection_reason": "disagreement"}
            for item in sorted(
                candidates,
                key=lambda item: (
                    item["local_jaccard"] + item["global_jaccard"],
                    -len(item["local_union"]),
                    -len(item["global_union"]),
                    item["image_id"],
                ),
            )[:limit]
        ]
    if mode == "agreement":
        return [
            {**item, "selection_reason": "agreement"}
            for item in sorted(
                candidates,
                key=lambda item: (
                    -item["local_jaccard"],
                    -item["global_jaccard"],
                    -len(item["local_intersection"]),
                    -item["total_abnormality_score"],
                    item["image_id"],
                ),
            )[:limit]
        ]
    if mode == "random":
        rng = random.Random(seed)
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        return [{**item, "selection_reason": "random"} for item in shuffled[:limit]]
    raise ValueError(f"unknown selection mode: {mode}")


def image_output_stem(rank: int, candidate: dict[str, Any]) -> str:
    return f"{rank:03d}_id_{candidate['image_id']}_{candidate['selection_reason']}"


def image_format_ext(image_format: str) -> str:
    return "jpg" if image_format.upper() == "JPEG" else image_format.lower()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def maybe_resize(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image, 1.0
    scale = max_side / float(current_max)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def scaled_box(box: list[float] | None, scale: float) -> tuple[float, float, float, float] | None:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) * scale for v in box]
    return x1, y1, x2, y2


def draw_dashed_line(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: tuple[int, int, int], width: int, dash: int) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        segment_end = min(pos + dash, length)
        if int(pos / dash) % 2 == 0:
            draw.line(
                [
                    (x1 + dx * pos, y1 + dy * pos),
                    (x1 + dx * segment_end, y1 + dy * segment_end),
                ],
                fill=fill,
                width=width,
            )
        pos += dash


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int,
    dashed: bool,
) -> None:
    x1, y1, x2, y2 = box
    if dashed:
        dash = max(8, width * 4)
        draw_dashed_line(draw, (x1, y1), (x2, y1), color, width, dash)
        draw_dashed_line(draw, (x2, y1), (x2, y2), color, width, dash)
        draw_dashed_line(draw, (x2, y2), (x1, y2), color, width, dash)
        draw_dashed_line(draw, (x1, y2), (x1, y1), color, width, dash)
    else:
        for offset in range(width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    return draw.textsize(text, font=font)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    text_w, text_h = text_size(draw, text, font)
    pad = 3
    y = max(0, y - text_h - pad * 2)
    draw.rectangle([x, y, x + text_w + pad * 2, y + text_h + pad * 2], fill=color)
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def draw_annotations(
    image: Image.Image,
    records: list[dict[str, Any]],
    labels: list[str],
    dashed_flags: list[bool],
    max_side: int,
) -> Image.Image:
    base = image.convert("RGB")
    base, scale = maybe_resize(base, max_side)
    draw = ImageDraw.Draw(base)
    width = max(2, round(max(base.size) / 500))
    font = load_font(max(12, round(max(base.size) / 90)))
    label_offsets: Counter = Counter()

    for record, annotator_label, dashed in zip(records, labels, dashed_flags):
        for index, item in enumerate(record.get("local_abnormalities", []), 1):
            box = scaled_box(item.get("bbox_xyxy"), scale)
            if box is None:
                continue
            code = item.get("code") or "L??"
            color = CODE_COLORS.get(code, (80, 80, 80))
            draw_box(draw, box, color, width=width, dashed=dashed)
            x1, y1, _, _ = box
            offset_key = (round(x1 / 30), round(y1 / 30))
            label_offsets[offset_key] += 1
            y_offset = (label_offsets[offset_key] - 1) * (font.size + 6 if hasattr(font, "size") else 18)
            severity = item.get("severity") or "?"
            label = f"{annotator_label}:{code}:{severity}:{index}"
            draw_label(draw, (x1, y1 + y_offset), label, color, font)
    return base


def copy_or_save_original(image: Image.Image, source_path: Path, destination: Path, max_side: int) -> None:
    if max_side <= 0:
        shutil.copy2(source_path, destination)
        return
    preview, _ = maybe_resize(image.convert("RGB"), max_side)
    preview.save(destination, quality=95)


def local_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(record.get("local_abnormalities", []), 1):
        rows.append(
            {
                "index": index,
                "code": item.get("code"),
                "name": LOCAL_CODE_NAMES.get(item.get("code"), item.get("label")),
                "severity": item.get("severity"),
                "severity_raw": item.get("severity_raw"),
                "bbox_xyxy": item.get("bbox_xyxy"),
            }
        )
    return rows


def global_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(record.get("global_abnormalities", []), 1):
        rows.append(
            {
                "index": index,
                "code": item.get("code"),
                "name": GLOBAL_CODE_NAMES.get(item.get("code"), item.get("label")),
                "severity": item.get("severity"),
                "severity_raw": item.get("severity_raw"),
            }
        )
    return rows


def render_annotation_md(candidate: dict[str, Any], output_files: dict[str, str]) -> str:
    a_record, b_record = candidate["records"]
    lines = [
        f"# image_id {candidate['image_id']}",
        "",
        f"- selection_reason: `{candidate['selection_reason']}`",
        f"- image_path: `{candidate.get('image_path')}`",
        f"- original_url: {candidate['image'].get('url')}",
        f"- A: `{a_record.get('record_key')}`, annotator `{a_record.get('annotator')}`",
        f"- B: `{b_record.get('record_key')}`, annotator `{b_record.get('annotator')}`",
        f"- validity: A `{a_record.get('validity')}`, B `{b_record.get('validity')}`",
        f"- local_jaccard: `{candidate['local_jaccard']}`",
        f"- global_jaccard: `{candidate['global_jaccard']}`",
        "",
        "## Files",
        "",
    ]
    for key, value in output_files.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## A Global", ""])
    lines.extend(summary_table(global_summary(a_record), ["index", "code", "name", "severity", "severity_raw"]))
    lines.extend(["", "## A Local", ""])
    lines.extend(summary_table(local_summary(a_record), ["index", "code", "name", "severity", "severity_raw", "bbox_xyxy"]))
    lines.extend(["", "## B Global", ""])
    lines.extend(summary_table(global_summary(b_record), ["index", "code", "name", "severity", "severity_raw"]))
    lines.extend(["", "## B Local", ""])
    lines.extend(summary_table(local_summary(b_record), ["index", "code", "name", "severity", "severity_raw", "bbox_xyxy"]))
    return "\n".join(lines) + "\n"


def summary_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["无"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(row.get(column)) for column in columns) + " |")
    return lines


def selection_row(candidate: dict[str, Any], rank: int, output_dir: Path) -> dict[str, Any]:
    return {
        "rank": rank,
        "image_id": candidate["image_id"],
        "selection_reason": candidate["selection_reason"],
        "image_path": candidate.get("image_path"),
        "output_dir": str(output_dir),
        "a_record_key": candidate.get("a_record_key"),
        "b_record_key": candidate.get("b_record_key"),
        "a_annotator": candidate.get("a_annotator"),
        "b_annotator": candidate.get("b_annotator"),
        "validity_a": candidate.get("validity_a"),
        "validity_b": candidate.get("validity_b"),
        "local_jaccard": candidate.get("local_jaccard"),
        "global_jaccard": candidate.get("global_jaccard"),
        "has_source_conflict": candidate.get("has_source_conflict"),
        "a_local_codes": candidate.get("a_local_codes"),
        "b_local_codes": candidate.get("b_local_codes"),
        "a_global_codes": candidate.get("a_global_codes"),
        "b_global_codes": candidate.get("b_global_codes"),
        "old_majority_label": candidate.get("old_majority_label"),
    }


def write_selection_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| rank | image_id | reason | source conflict | A | B | validity A/B | local J | global J | A local | B local | A global | B global |",
        "| ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(row["rank"]),
                    md_cell(row["image_id"]),
                    md_cell(row["selection_reason"]),
                    md_cell(row["has_source_conflict"]),
                    md_cell(row["a_annotator"]),
                    md_cell(row["b_annotator"]),
                    md_cell(f"{row['validity_a']} / {row['validity_b']}"),
                    md_cell(row["local_jaccard"]),
                    md_cell(row["global_jaccard"]),
                    md_cell(",".join(row["a_local_codes"])),
                    md_cell(",".join(row["b_local_codes"])),
                    md_cell(",".join(row["a_global_codes"])),
                    md_cell(",".join(row["b_global_codes"])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_selection(
    selected: list[dict[str, Any]],
    output_dir: Path,
    max_side: int,
    image_format: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []

    for rank, candidate in enumerate(selected, 1):
        source_path_text = candidate.get("image_path")
        if not source_path_text:
            print(f"skip image_id={candidate['image_id']}: image file not found", file=sys.stderr)
            continue
        source_path = Path(source_path_text)
        stem = image_output_stem(rank, candidate)
        image_dir = output_dir / stem
        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(source_path)
            image.load()
        except (OSError, UnidentifiedImageError) as exc:
            print(f"skip image_id={candidate['image_id']}: cannot open image: {exc}", file=sys.stderr)
            continue

        format_ext = image_format_ext(image_format)
        original_suffix = source_path.suffix.lower() if max_side <= 0 and source_path.suffix else f".{format_ext}"
        original_path = image_dir / f"original{original_suffix}"
        a_path = image_dir / f"bbox_A.{format_ext}"
        b_path = image_dir / f"bbox_B.{format_ext}"
        ab_path = image_dir / f"bbox_A_B.{format_ext}"
        annotations_json_path = image_dir / "annotations.json"
        annotations_md_path = image_dir / "annotations.md"

        copy_or_save_original(image, source_path, original_path, max_side)
        a_image = draw_annotations(image, [candidate["records"][0]], ["A"], [False], max_side)
        b_image = draw_annotations(image, [candidate["records"][1]], ["B"], [False], max_side)
        ab_image = draw_annotations(image, candidate["records"], ["A", "B"], [False, True], max_side)
        a_image.save(a_path, quality=95)
        b_image.save(b_path, quality=95)
        ab_image.save(ab_path, quality=95)

        output_files = {
            "original": original_path.name,
            "bbox_A": a_path.name,
            "bbox_B": b_path.name,
            "bbox_A_B": ab_path.name,
            "annotations_json": annotations_json_path.name,
            "annotations_md": annotations_md_path.name,
        }
        annotation_payload = {
            "image_id": candidate["image_id"],
            "selection_reason": candidate["selection_reason"],
            "image": candidate["image"],
            "image_path": candidate.get("image_path"),
            "metrics": {
                "validity_agree": candidate["validity_agree"],
                "local_jaccard": candidate["local_jaccard"],
                "global_jaccard": candidate["global_jaccard"],
                "local_intersection": candidate["local_intersection"],
                "global_intersection": candidate["global_intersection"],
                "local_union": candidate["local_union"],
                "global_union": candidate["global_union"],
                "total_local_boxes": candidate["total_local_boxes"],
                "total_global_labels": candidate["total_global_labels"],
            },
            "annotator_A": {
                "record": candidate["records"][0],
                "local_summary": local_summary(candidate["records"][0]),
                "global_summary": global_summary(candidate["records"][0]),
            },
            "annotator_B": {
                "record": candidate["records"][1],
                "local_summary": local_summary(candidate["records"][1]),
                "global_summary": global_summary(candidate["records"][1]),
            },
            "output_files": output_files,
        }
        write_json(annotations_json_path, annotation_payload)
        annotations_md_path.write_text(render_annotation_md(candidate, output_files), encoding="utf-8")

        selection_rows.append(selection_row(candidate, rank, image_dir))

    write_jsonl(output_dir / "selected_images.jsonl", selection_rows)
    write_selection_markdown(output_dir / "selected_images.md", selection_rows, "Selected Annotation Visualizations")
    return selection_rows


def parse_id_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else dataset_root / DEFAULT_ANALYSIS_SUBDIR
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = analysis_dir / f"visualizations_{args.selection}_{args.limit}"

    images, records_by_image = load_inputs(analysis_dir)
    candidates = build_candidates(images, records_by_image, dataset_root, args.include_missing_images)
    if not candidates:
        raise RuntimeError(f"no candidates found from {analysis_dir}")

    selected = select_candidates(candidates, args.limit, args.selection, args.seed, parse_id_list(args.image_ids))
    rows = visualize_selection(selected, output_dir, args.max_side, args.image_format)
    print(f"analysis_dir={analysis_dir}")
    print(f"output_dir={output_dir}")
    print(f"candidate_images={len(candidates)} selected_images={len(rows)}")
    print(f"selection_summary={output_dir / 'selected_images.md'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select and visualize A/B local bbox annotations for the 1k dataset.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help=f"Directory produced by prepare_1k_annotations.py. Defaults to dataset-root/{DEFAULT_ANALYSIS_SUBDIR}.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Number of images to visualize.")
    parser.add_argument(
        "--selection",
        choices=["balanced", "disagreement", "agreement", "random"],
        default="balanced",
        help="Image selection strategy.",
    )
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--image-ids", default=None, help="Comma-separated explicit image ids; overrides --selection.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize visual outputs so max side is at most this value. 0 keeps original size.",
    )
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--include-missing-images", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
