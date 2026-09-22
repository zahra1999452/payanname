# -*- coding: utf-8 -*-

"""
===============================================================================
H2 ROBUSTNESS EVALUATION
CACHED / RESUMABLE / IMAGE-ID SAFE VERSION
===============================================================================

Purpose
-------
Evaluate YOLOv8, RT-DETR and RTMDet on:

1. Clean test images
2. brightness
3. contrast
4. gaussian_noise
5. gaussian_blur
6. motion_blur
7. jpeg_compression

The script:
- validates image IDs using normalized Path.stem
- does NOT compare filename extensions
- does NOT depend on file ordering
- uses clean YOLO labels for both clean/corrupted images
- converts YOLO normalized GT boxes to pixel coordinates
- performs class-aware greedy matching
- calculates per-image Precision / Recall / F1 / IoU
- caches predictions
- resumes interrupted runs
- supports --force
- supports --limit
- produces CSV/JSON outputs
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]

CORRUPTIONS = [
    "brightness",
    "contrast",
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "jpeg_compression",
]

ALL_CONDITIONS = ["clean"] + CORRUPTIONS

CACHE_VERSION = "H2_CACHE_V3_IMAGE_ID_SAFE"

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_DEVICE = "cuda:0"

# Expected class range for the final taxonomy.
# If your actual experiment uses 13 classes => 0..12.
NUM_CLASSES = 13

# Cache is written every N images.
CACHE_SAVE_EVERY = 20


# =============================================================================
# PRINT HELPERS
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


# =============================================================================
# PATH / STRING HELPERS
# =============================================================================

def normalize_image_id(value: Any) -> str:
    """
    Normalize an image identifier.

    Important:
    We intentionally use only the filename stem.
    Therefore:

        abc.jpg
        abc.JPG
        abc.png

    all become:

        abc
    """

    if value is None:
        return ""

    s = str(value).strip()

    # Remove directory if accidentally supplied.
    s = Path(s).name

    # Remove extension.
    s = Path(s).stem

    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s).strip()

    # Case-insensitive comparison.
    return s.lower()


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def condition_slug(condition: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", condition.lower()).strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def list_images(directory: Path) -> List[Path]:
    """
    Return all valid image files recursively/directly.

    For the experiment we normally expect files directly inside the directory.
    rglob is used to be robust to nested folders.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{directory}"
        )

    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    return sorted(files, key=lambda p: normalize_image_id(p.name))


def build_image_map(
    directory: Path,
    description: str,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    Build:

        image_id -> path

    and also detect duplicate normalized IDs.
    """

    files = list_images(directory)

    mapping: Dict[str, Path] = defaultdict()
    duplicates: Dict[str, List[Path]] = defaultdict(list)

    for p in files:
        image_id = normalize_image_id(p.name)

        if not image_id:
            continue

        if image_id in mapping:
            duplicates[image_id].append(p)
        else:
            mapping[image_id] = p

    if duplicates:
        fail(f"Duplicate normalized image IDs found in {description}:")
        for image_id, paths in list(duplicates.items())[:20]:
            print(f"  ID: {image_id}")
            print(f"     {mapping[image_id]}")
            for p in paths:
                print(f"     {p}")

        raise RuntimeError(
            f"Duplicate image IDs detected in {description}."
        )

    return dict(mapping), dict(duplicates)


def resolve_corruption_images_dir(
    corruption_root: Path,
    corruption_name: str,
) -> Path:
    """
    Expected structure:

    corruption/
        brightness/
            images/
        contrast/
            images/
        ...

    We also support a fallback where images are directly inside the
    corruption directory.
    """

    corruption_dir = corruption_root / corruption_name

    if not corruption_dir.exists():
        raise FileNotFoundError(
            f"Corruption directory does not exist:\n{corruption_dir}"
        )

    images_dir = corruption_dir / "images"

    if images_dir.exists() and images_dir.is_dir():
        return images_dir

    # Fallback.
    direct_images = [
        p
        for p in corruption_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if direct_images:
        return corruption_dir

    raise FileNotFoundError(
        f"No images found for corruption '{corruption_name}'.\n"
        f"Expected:\n{images_dir}\n"
        f"or images directly under:\n{corruption_dir}"
    )


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_corruption_images(
    clean_map: Dict[str, Path],
    corruption_map: Dict[str, Path],
    corruption_name: str,
) -> Dict[str, Any]:
    """
    IMPORTANT FIX

    Compare normalized image IDs only.

    The old implementation incorrectly produced:

        Missing: 2462

    even though the filenames were actually identical.

    This implementation does not compare Path objects, extensions,
    ordering or absolute paths.
    """

    clean_ids = set(clean_map.keys())
    corruption_ids = set(corruption_map.keys())

    missing = sorted(clean_ids - corruption_ids)
    extra = sorted(corruption_ids - clean_ids)

    result = {
        "condition": corruption_name,
        "clean_count": len(clean_ids),
        "corruption_count": len(corruption_ids),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "status": "PASS",
    }

    if missing or extra:
        result["status"] = "FAIL"

        fail(f"{corruption_name}")
        print(f"  Clean IDs       : {len(clean_ids)}")
        print(f"  Corruption IDs  : {len(corruption_ids)}")
        print(f"  Missing         : {len(missing)}")
        print(f"  Extra           : {len(extra)}")

        if missing:
            print(
                "  Missing examples:",
                missing[:20],
            )

        if extra:
            print(
                "  Extra examples:",
                extra[:20],
            )

        raise RuntimeError(
            f"Image-ID mismatch for corruption: {corruption_name}"
        )

    ok(
        f"{corruption_name}: "
        f"{len(clean_ids)} IDs matched exactly"
    )

    return result


def validate_all_corruptions(
    clean_map: Dict[str, Path],
    corruption_root: Path,
) -> Tuple[Dict[str, Dict[str, Path]], List[Dict[str, Any]]]:

    banner("CORRUPTION IMAGE-ID VALIDATION")

    corruption_maps: Dict[str, Dict[str, Path]] = {}
    validation_results: List[Dict[str, Any]] = []

    for corruption in CORRUPTIONS:

        images_dir = resolve_corruption_images_dir(
            corruption_root,
            corruption,
        )

        info(
            f"{corruption}: {images_dir}"
        )

        corruption_map, _ = build_image_map(
            images_dir,
            f"corruption/{corruption}",
        )

        result = validate_corruption_images(
            clean_map,
            corruption_map,
            corruption,
        )

        corruption_maps[corruption] = corruption_map
        validation_results.append(result)

    return corruption_maps, validation_results


# =============================================================================
# IMAGE SIZE
# =============================================================================

def get_image_size(image_path: Path) -> Tuple[int, int]:
    """
    Returns:
        width, height
    """

    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return int(im.width), int(im.height)

    except Exception as exc:
        raise RuntimeError(
            f"Could not read image size:\n{image_path}\n{exc}"
        ) from exc


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_yolo_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Read YOLO format:

        class_id cx cy w h

    normalized to [0,1]

    and convert to absolute pixel xyxy:

        x1 y1 x2 y2
    """

    gt: List[Dict[str, Any]] = []

    if not label_path.exists():
        # Missing label = zero GT objects.
        # It is recorded later in integrity information.
        return gt

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        text = label_path.read_text(
            encoding="latin-1",
            errors="ignore",
        )

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            warn(
                f"Invalid label line ignored: "
                f"{label_path} line {line_number}"
            )
            continue

        try:
            class_id = int(float(parts[0]))

            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except ValueError:
            warn(
                f"Non-numeric label ignored: "
                f"{label_path} line {line_number}"
            )
            continue

        # Clamp to valid normalized range.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = max(bw, 0.0)
        bh = max(bh, 0.0)

        x_center = cx * image_width
        y_center = cy * image_height
        box_width = bw * image_width
        box_height = bh * image_height

        x1 = x_center - box_width / 2.0
        y1 = y_center - box_height / 2.0
        x2 = x_center + box_width / 2.0
        y2 = y_center + box_height / 2.0

        # Clamp absolute coordinates.
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))

        # Ignore invalid/zero-area boxes.
        if x2 <= x1 or y2 <= y1:
            continue

        gt.append(
            {
                "class_id": class_id,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
            }
        )

    return gt


# =============================================================================
# IOU
# =============================================================================

def calculate_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


# =============================================================================
# PREDICTION MATCHING
# =============================================================================

def match_predictions(
    gt: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    """
    Class-aware greedy matching.

    Each GT and prediction can be matched at most once.
    """

    if not gt and not predictions:
        return {
            "matched": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": [],
        }

    if not gt:
        return {
            "matched": 0,
            "tp": 0,
            "fp": len(predictions),
            "fn": 0,
            "ious": [],
        }

    if not predictions:
        return {
            "matched": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt),
            "ious": [],
        }

    candidate_matches = []

    for gi, gt_item in enumerate(gt):

        gt_class = int(gt_item["class_id"])
        gt_box = gt_item["bbox"]

        for pi, pred_item in enumerate(predictions):

            pred_class = int(pred_item["class_id"])

            if gt_class != pred_class:
                continue

            iou = calculate_iou(
                gt_box,
                pred_item["bbox"],
            )

            if iou >= iou_threshold:
                candidate_matches.append(
                    (
                        float(iou),
                        gi,
                        pi,
                    )
                )

    # Highest IoU first.
    candidate_matches.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    for iou, gi, pi in candidate_matches:

        if gi in matched_gt:
            continue

        if pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_ious.append(float(iou))

    tp = len(matched_ious)
    fp = len(predictions) - tp
    fn = len(gt) - tp

    return {
        "matched": tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ious": matched_ious,
    }


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics_from_match(
    gt_count: int,
    prediction_count: int,
    match_result: Dict[str, Any],
) -> Dict[str, Any]:

    tp = int(match_result["tp"])
    fp = int(match_result["fp"])
    fn = int(match_result["fn"])

    # Precision
    if prediction_count == 0:
        if gt_count == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp / prediction_count

    # Recall
    if gt_count == 0:
        if prediction_count == 0:
            recall = 1.0
        else:
            # No GT exists. Recall is not naturally defined.
            # For paired F1 evaluation we treat it as 1.
            recall = 1.0
    else:
        recall = tp / gt_count

    # F1
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )

    ious = match_result.get("ious", [])

    if ious:
        mean_iou = float(np.mean(ious))
    else:
        mean_iou = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "gt_count": int(gt_count),
        "prediction_count": int(prediction_count),
        "matched_count": int(tp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# =============================================================================
# MODEL IMPORTS
# =============================================================================

def import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as exc:
        raise ImportError(
            "Ultralytics could not be imported.\n"
            "Install/use the environment where YOLOv8 and RT-DETR were trained.\n"
            f"Original error: {exc}"
        ) from exc


def import_mmdetection():
    try:
        from mmdet.apis import init_detector, inference_detector
        return init_detector, inference_detector
    except Exception as exc:
        raise ImportError(
            "MMDetection could not be imported.\n"
            f"Original error: {exc}"
        ) from exc


# =============================================================================
# YOLOv8 / RT-DETR PREDICTION
# =============================================================================

def predict_ultralytics(
    model: Any,
    image_path: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Works for both:
        YOLOv8
        RT-DETR

    using Ultralytics API.
    """

    results = model.predict(
        source=str(image_path),
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    boxes = result.boxes

    if boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()

    if boxes.conf is not None:
        scores = boxes.conf.detach().cpu().numpy()
    else:
        scores = np.ones(
            len(xyxy),
            dtype=np.float32,
        )

    if boxes.cls is not None:
        classes = boxes.cls.detach().cpu().numpy()
    else:
        classes = np.zeros(
            len(xyxy),
            dtype=np.float32,
        )

    predictions = []

    for box, score, cls_id in zip(
        xyxy,
        scores,
        classes,
    ):

        score = float(score)

        if score < conf:
            continue

        x1, y1, x2, y2 = [
            float(v)
            for v in box[:4]
        ]

        if x2 <= x1 or y2 <= y1:
            continue

        predictions.append(
            {
                "class_id": int(cls_id),
                "confidence": score,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

    return predictions


# =============================================================================
# RTMDET PREDICTION
# =============================================================================

def predict_rtmdet(
    model: Any,
    inference_detector: Any,
    image_path: Path,
    conf: float,
) -> List[Dict[str, Any]]:

    result = inference_detector(
        model,
        str(image_path),
    )

    predictions = []

    # -------------------------------------------------------------------------
    # MMDetection 3.x
    # -------------------------------------------------------------------------

    if hasattr(result, "pred_instances"):

        pred = result.pred_instances

        if pred is None:
            return []

        if hasattr(pred, "bboxes"):
            bboxes = pred.bboxes.detach().cpu().numpy()
        else:
            bboxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )

        if hasattr(pred, "scores"):
            scores = pred.scores.detach().cpu().numpy()
        else:
            scores = np.ones(
                len(bboxes),
                dtype=np.float32,
            )

        if hasattr(pred, "labels"):
            labels = pred.labels.detach().cpu().numpy()
        else:
            labels = np.zeros(
                len(bboxes),
                dtype=np.int64,
            )

        for box, score, label in zip(
            bboxes,
            scores,
            labels,
        ):

            score = float(score)

            if score < conf:
                continue

            x1, y1, x2, y2 = [
                float(v)
                for v in box[:4]
            ]

            if x2 <= x1 or y2 <= y1:
                continue

            predictions.append(
                {
                    "class_id": int(label),
                    "confidence": score,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

        return predictions

    # -------------------------------------------------------------------------
    # Older MMDetection style
    # -------------------------------------------------------------------------

    if isinstance(result, (list, tuple)):

        for class_id, class_result in enumerate(result):

            if class_result is None:
                continue

            arr = np.asarray(class_result)

            if arr.size == 0:
                continue

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            for row in arr:

                if len(row) < 4:
                    continue

                x1, y1, x2, y2 = [
                    float(v)
                    for v in row[:4]
                ]

                score = (
                    float(row[4])
                    if len(row) >= 5
                    else 1.0
                )

                if score < conf:
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                predictions.append(
                    {
                        "class_id": int(class_id),
                        "confidence": score,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

        return predictions

    return predictions


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(
    args: argparse.Namespace,
) -> Dict[str, Any]:

    banner("MODEL LOADING")

    models: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Ultralytics
    # -------------------------------------------------------------------------

    YOLO = import_ultralytics()

    info(f"Loading YOLOv8: {args.yolov8}")
    models["YOLOv8"] = {
        "type": "ultralytics",
        "model": YOLO(str(args.yolov8)),
    }
    ok("YOLOv8 loaded")

    info(f"Loading RT-DETR: {args.rtdetr}")
    models["RT-DETR"] = {
        "type": "ultralytics",
        "model": YOLO(str(args.rtdetr)),
    }
    ok("RT-DETR loaded")

    # -------------------------------------------------------------------------
    # RTMDet
    # -------------------------------------------------------------------------

    init_detector, inference_detector = import_mmdetection()

    info(f"Loading RTMDet config: {args.rtmdet_config}")
    info(f"Loading RTMDet checkpoint: {args.rtmdet}")

    rtmdet_model = init_detector(
        str(args.rtmdet_config),
        str(args.rtmdet),
        device=args.device,
    )

    models["RTMDet"] = {
        "type": "mmdet",
        "model": rtmdet_model,
        "inference_detector": inference_detector,
    }

    ok("RTMDet loaded")

    return models


# =============================================================================
# CACHE
# =============================================================================

def cache_file_path(
    cache_dir: Path,
    condition: str,
    model_name: str,
) -> Path:

    filename = (
        f"{condition_slug(condition)}"
        f"__"
        f"{model_slug(model_name)}"
        f"__predictions.json"
    )

    return cache_dir / filename


def cache_metadata(
    args: argparse.Namespace,
    condition: str,
    model_name: str,
) -> Dict[str, Any]:

    return {
        "cache_version": CACHE_VERSION,
        "model": model_name,
        "condition": condition,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "device": str(args.device),
        "model_path": str(
            getattr(args, model_name.lower().replace("-", ""), "")
        ),
    }


def load_prediction_cache(
    path: Path,
    expected_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Load cache safely.

    Returns:
        image_id -> record
    """

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except Exception as exc:
        warn(
            f"Could not read cache {path.name}: {exc}"
        )
        return {}

    if not isinstance(data, dict):
        return {}

    metadata = data.get("metadata", {})

    # Verify important parameters.
    important_keys = [
        "cache_version",
        "model",
        "condition",
        "imgsz",
        "conf",
        "iou",
        "device",
    ]

    for key in important_keys:

        if metadata.get(key) != expected_metadata.get(key):
            warn(
                f"Cache metadata mismatch: {path.name}"
            )
            return {}

    records = data.get("records", [])

    if not isinstance(records, list):
        return {}

    result = {}

    for record in records:

        if not isinstance(record, dict):
            continue

        image_id = normalize_image_id(
            record.get("image_id", "")
        )

        if not image_id:
            continue

        predictions = record.get(
            "predictions",
            [],
        )

        if not isinstance(predictions, list):
            continue

        result[image_id] = {
            "image_id": image_id,
            "predictions": predictions,
        }

    return result


def save_prediction_cache(
    path: Path,
    metadata: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
) -> None:

    ensure_dir(path.parent)

    ordered_records = [
        records[k]
        for k in sorted(records.keys())
    ]

    payload = {
        "metadata": metadata,
        "record_count": len(ordered_records),
        "records": ordered_records,
    }

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Atomic replacement.
    temp_path.replace(path)


# =============================================================================
# INFERENCE CACHE BUILDER
# =============================================================================

def run_or_resume_inference(
    model_name: str,
    condition: str,
    image_map: Dict[str, Path],
    models: Dict[str, Any],
    args: argparse.Namespace,
    cache_dir: Path,
) -> Dict[str, Dict[str, Any]]:

    cache_path = cache_file_path(
        cache_dir,
        condition,
        model_name,
    )

    metadata = cache_metadata(
        args,
        condition,
        model_name,
    )

    records = load_prediction_cache(
        cache_path,
        metadata,
    )

    all_ids = sorted(image_map.keys())

    # Apply limit only for the current run.
    if args.limit is not None:
        selected_ids = all_ids[:args.limit]
    else:
        selected_ids = all_ids

    missing_ids = [
        image_id
        for image_id in selected_ids
        if image_id not in records
    ]

    if not missing_ids:
        print(
            f"[CACHE HIT] {model_name} / {condition} "
            f"({len(selected_ids)} images)"
        )
        return {
            image_id: records[image_id]
            for image_id in selected_ids
        }

    print()
    print(
        f"[CACHE MISS] {model_name} / {condition}"
    )
    print(
        f"Already cached : {len(records)}"
    )
    print(
        f"Need inference : {len(missing_ids)}"
    )

    model_info = models[model_name]

    processed_since_save = 0

    for index, image_id in enumerate(
        missing_ids,
        start=1,
    ):

        image_path = image_map[image_id]

        try:

            if model_info["type"] == "ultralytics":

                predictions = predict_ultralytics(
                    model_info["model"],
                    image_path,
                    args.imgsz,
                    args.conf,
                    args.device,
                )

            elif model_info["type"] == "mmdet":

                predictions = predict_rtmdet(
                    model_info["model"],
                    model_info["inference_detector"],
                    image_path,
                    args.conf,
                )

            else:
                raise RuntimeError(
                    f"Unknown model type: "
                    f"{model_info['type']}"
                )

            records[image_id] = {
                "image_id": image_id,
                "predictions": predictions,
            }

            processed_since_save += 1

            if (
                processed_since_save >= CACHE_SAVE_EVERY
                or index == len(missing_ids)
            ):

                save_prediction_cache(
                    cache_path,
                    metadata,
                    records,
                )

                processed_since_save = 0

            if (
                index == 1
                or index % 25 == 0
                or index == len(missing_ids)
            ):

                print(
                    f"  [{index:4d}/{len(missing_ids)}] "
                    f"{image_id}"
                )

        except KeyboardInterrupt:

            warn(
                "KeyboardInterrupt detected. "
                "Saving current cache..."
            )

            save_prediction_cache(
                cache_path,
                metadata,
                records,
            )

            raise

        except Exception as exc:

            fail(
                f"{model_name} / {condition} / "
                f"{image_id}"
            )

            print(
                f"  Image: {image_path}"
            )
            print(
                f"  Error: {exc}"
            )

            # Do NOT write failed inference to cache.
            # It will be retried in the next run.

    # Final save.
    save_prediction_cache(
        cache_path,
        metadata,
        records,
    )

    # Return only selected IDs.
    return {
        image_id: records[image_id]
        for image_id in selected_ids
        if image_id in records
    }


# =============================================================================
# PER-IMAGE EVALUATION
# =============================================================================

def evaluate_condition_model(
    model_name: str,
    condition: str,
    image_map: Dict[str, Path],
    prediction_records: Dict[str, Dict[str, Any]],
    labels_dir: Path,
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Evaluate cached predictions against YOLO labels.

    IMPORTANT:
    GT coordinates are converted to pixel coordinates using the actual
    image width/height.
    """

    rows = []

    for image_id in sorted(image_map.keys()):

        image_path = image_map[image_id]

        if image_id not in prediction_records:
            # Missing inference result.
            continue

        record = prediction_records[image_id]

        predictions = record.get(
            "predictions",
            [],
        )

        width, height = get_image_size(
            image_path
        )

        label_path = labels_dir / f"{image_id}.txt"

        gt = load_yolo_ground_truth(
            label_path,
            width,
            height,
        )

        match = match_predictions(
            gt,
            predictions,
            iou_threshold,
        )

        metrics = calculate_metrics_from_match(
            len(gt),
            len(predictions),
            match,
        )

        row = {
            "image_id": image_id,
            "model": model_name,
            "condition": condition,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "image_width": width,
            "image_height": height,
            "label_exists": bool(label_path.exists()),
            **metrics,
        }

        rows.append(row)

    return rows


# =============================================================================
# DEGRADATION
# =============================================================================

def calculate_degradation(
    clean_f1: float,
    corrupted_f1: float,
) -> Tuple[float, float, float]:
    """
    Returns:

        relative_drop_percent
        absolute_drop
        retention_percent

    Formula:

        Relative Drop =
            (Clean F1 - Corrupted F1) / Clean F1 * 100

    when Clean F1 > 0.

    If Clean F1 == 0:
        relative drop is not mathematically defined.
        We store NaN.

    Retention:

        Corrupted F1 / Clean F1 * 100

    when Clean F1 > 0.
    """

    absolute_drop = clean_f1 - corrupted_f1

    if clean_f1 > 0:

        relative_drop = (
            absolute_drop
            / clean_f1
            * 100.0
        )

        retention = (
            corrupted_f1
            / clean_f1
            * 100.0
        )

    else:

        relative_drop = np.nan
        retention = np.nan

    return (
        float(relative_drop)
        if not math.isnan(relative_drop)
        else np.nan,
        float(absolute_drop),
        float(retention)
        if not math.isnan(retention)
        else np.nan,
    )


def build_degradation_dataframe(
    per_image_df: pd.DataFrame,
) -> pd.DataFrame:

    clean = per_image_df[
        per_image_df["condition"] == "clean"
    ][
        [
            "image_id",
            "model",
            "f1",
            "precision",
            "recall",
            "mean_iou",
        ]
    ].copy()

    clean = clean.rename(
        columns={
            "f1": "clean_f1",
            "precision": "clean_precision",
            "recall": "clean_recall",
            "mean_iou": "clean_mean_iou",
        }
    )

    corrupted = per_image_df[
        per_image_df["condition"] != "clean"
    ][
        [
            "image_id",
            "model",
            "condition",
            "f1",
            "precision",
            "recall",
            "mean_iou",
        ]
    ].copy()

    corrupted = corrupted.rename(
        columns={
            "f1": "corrupted_f1",
            "precision": "corrupted_precision",
            "recall": "corrupted_recall",
            "mean_iou": "corrupted_mean_iou",
        }
    )

    merged = corrupted.merge(
        clean,
        on=[
            "image_id",
            "model",
        ],
        how="left",
        validate="many_to_one",
    )

    degradation_values = merged.apply(
        lambda row: calculate_degradation(
            float(row["clean_f1"]),
            float(row["corrupted_f1"]),
        ),
        axis=1,
        result_type="expand",
    )

    degradation_values.columns = [
        "relative_drop_percent",
        "absolute_f1_drop",
        "retention_percent",
    ]

    merged = pd.concat(
        [
            merged.reset_index(drop=True),
            degradation_values,
        ],
        axis=1,
    )

    return merged


# =============================================================================
# SUMMARY
# =============================================================================

def build_h2_summary(
    degradation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = degradation_df.groupby(
        [
            "model",
            "condition",
        ],
        dropna=False,
    )

    for (model, condition), group in grouped:

        f1 = group["corrupted_f1"].astype(float)
        clean_f1 = group["clean_f1"].astype(float)

        degradation = group[
            "relative_drop_percent"
        ].astype(float)

        absolute_drop = group[
            "absolute_f1_drop"
        ].astype(float)

        retention = group[
            "retention_percent"
        ].astype(float)

        rows.append(
            {
                "model": model,
                "condition": condition,
                "n_pairs": int(len(group)),

                "mean_clean_f1": float(
                    clean_f1.mean()
                ),

                "mean_corrupted_f1": float(
                    f1.mean()
                ),

                "std_corrupted_f1": float(
                    f1.std(ddof=1)
                    if len(f1) > 1
                    else 0.0
                ),

                "median_corrupted_f1": float(
                    f1.median()
                ),

                "q1_corrupted_f1": float(
                    f1.quantile(0.25)
                ),

                "q3_corrupted_f1": float(
                    f1.quantile(0.75)
                ),

                "min_corrupted_f1": float(
                    f1.min()
                ),

                "max_corrupted_f1": float(
                    f1.max()
                ),

                "mean_relative_drop_percent": float(
                    degradation.mean()
                )
                if degradation.notna().any()
                else np.nan,

                "median_relative_drop_percent": float(
                    degradation.median()
                )
                if degradation.notna().any()
                else np.nan,

                "std_relative_drop_percent": float(
                    degradation.std(ddof=1)
                    if degradation.notna().sum() > 1
                    else 0.0
                )
                if degradation.notna().any()
                else np.nan,

                "mean_absolute_f1_drop": float(
                    absolute_drop.mean()
                ),

                "median_absolute_f1_drop": float(
                    absolute_drop.median()
                ),

                "mean_retention_percent": float(
                    retention.mean()
                )
                if retention.notna().any()
                else np.nan,

                "median_retention_percent": float(
                    retention.median()
                )
                if retention.notna().any()
                else np.nan,

                "n_clean_f1_zero": int(
                    (clean_f1 == 0).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CONDITION COMPARISON
# =============================================================================

def build_condition_wins(
    degradation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive condition comparison.

    For each model, count which corruption has the lowest mean degradation.
    This is NOT a statistical hypothesis test.
    """

    rows = []

    summary = (
        degradation_df
        .groupby(
            ["model", "condition"],
            dropna=False,
        )[
            "relative_drop_percent"
        ]
        .mean()
        .reset_index()
    )

    for model in MODELS:

        subset = summary[
            summary["model"] == model
        ].copy()

        subset = subset[
            subset["condition"] != "clean"
        ]

        if subset.empty:
            continue

        subset = subset.sort_values(
            "relative_drop_percent",
            ascending=True,
        )

        for rank, (_, row) in enumerate(
            subset.iterrows(),
            start=1,
        ):

            rows.append(
                {
                    "model": model,
                    "condition": row["condition"],
                    "mean_relative_drop_percent":
                        row["relative_drop_percent"],
                    "descriptive_rank_within_model":
                        rank,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# INTEGRITY
# =============================================================================

def build_integrity_report(
    per_image_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    clean_count: int,
    corruption_counts: Dict[str, int],
    limit: Optional[int],
) -> Dict[str, Any]:

    expected_image_count = (
        limit
        if limit is not None
        else clean_count
    )

    expected_clean_rows = (
        expected_image_count
        * len(MODELS)
    )

    expected_corrupted_rows = (
        expected_image_count
        * len(CORRUPTIONS)
        * len(MODELS)
    )

    expected_degradation_rows = (
        expected_image_count
        * len(CORRUPTIONS)
        * len(MODELS)
    )

    actual_clean_rows = int(
        (
            per_image_df["condition"] == "clean"
        ).sum()
    )

    actual_corrupted_rows = int(
        (
            per_image_df["condition"] != "clean"
        ).sum()
    )

    actual_degradation_rows = int(
        len(degradation_df)
    )

    missing_label_count = int(
        (
            per_image_df["label_exists"] == False
        ).sum()
    )

    nan_f1_count = int(
        per_image_df["f1"].isna().sum()
    )

    nan_model_count = int(
        per_image_df["model"].isna().sum()
    )

    nan_condition_count = int(
        per_image_df["condition"].isna().sum()
    )

    clean_pairs = set(
        zip(
            per_image_df.loc[
                per_image_df["condition"] == "clean",
                "image_id",
            ],
            per_image_df.loc[
                per_image_df["condition"] == "clean",
                "model",
            ],
        )
    )

    corruption_pairs = set(
        zip(
            degradation_df["image_id"],
            degradation_df["model"],
        )
    )

    missing_degradation_pairs = sorted(
        clean_pairs - corruption_pairs
    )

    report = {
        "status": "PASS",

        "expected": {
            "clean_images": expected_image_count,
            "models": len(MODELS),
            "corruptions": len(CORRUPTIONS),
            "clean_rows": expected_clean_rows,
            "corrupted_rows": expected_corrupted_rows,
            "degradation_rows": expected_degradation_rows,
        },

        "actual": {
            "clean_rows": actual_clean_rows,
            "corrupted_rows": actual_corrupted_rows,
            "degradation_rows": actual_degradation_rows,
        },

        "corruption_image_counts": corruption_counts,

        "quality": {
            "missing_label_count": missing_label_count,
            "nan_f1_count": nan_f1_count,
            "nan_model_count": nan_model_count,
            "nan_condition_count": nan_condition_count,
            "missing_degradation_pairs":
                len(missing_degradation_pairs),
            "missing_degradation_examples":
                missing_degradation_pairs[:20],
        },

        "complete": (
            actual_clean_rows
            == expected_clean_rows
            and
            actual_corrupted_rows
            == expected_corrupted_rows
            and
            actual_degradation_rows
            == expected_degradation_rows
            and
            len(missing_degradation_pairs) == 0
            and
            nan_f1_count == 0
            and
            nan_model_count == 0
            and
            nan_condition_count == 0
        ),
    }

    if not report["complete"]:
        report["status"] = "FAIL"

    return report


# =============================================================================
# JSON UTILITIES
# =============================================================================

def json_default(obj: Any) -> Any:

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj)} is not JSON serializable"
    )


def save_json(
    path: Path,
    data: Any,
) -> None:

    ensure_dir(path.parent)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

def build_run_configuration(
    args: argparse.Namespace,
) -> Dict[str, Any]:

    return {
        "cache_version": CACHE_VERSION,

        "test_images": str(
            Path(args.test_images).resolve()
        ),

        "test_labels": str(
            Path(args.test_labels).resolve()
        ),

        "corruption_root": str(
            Path(args.corruption_root).resolve()
        ),

        "models": {
            "YOLOv8": str(
                Path(args.yolov8).resolve()
            ),
            "RT-DETR": str(
                Path(args.rtdetr).resolve()
            ),
            "RTMDet": str(
                Path(args.rtmdet).resolve()
            ),
            "RTMDet_config": str(
                Path(args.rtmdet_config).resolve()
            ),
        },

        "output": str(
            Path(args.output).resolve()
        ),

        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "device": str(args.device),

        "models_evaluated": MODELS,
        "corruptions_evaluated": CORRUPTIONS,

        "limit": (
            int(args.limit)
            if args.limit is not None
            else None
        ),
    }


# =============================================================================
# FORCE CLEAN
# =============================================================================

def handle_force(
    output_dir: Path,
) -> None:

    cache_dir = output_dir / "cache"

    if cache_dir.exists():
        print(
            f"[FORCE] Removing cache directory:\n"
            f"{cache_dir}"
        )

        shutil.rmtree(cache_dir)

    # Remove generated result files.
    files_to_remove = [
        "clean_per_image_metrics.csv",
        "corrupted_per_image_metrics.csv",
        "per_image_metrics.csv",
        "per_image_degradation.csv",
        "H2_degradation_summary.csv",
        "H2_condition_wins.csv",
        "H2_integrity_check.json",
        "H2_image_id_validation.json",
        "H2_run_configuration.json",
    ]

    for filename in files_to_remove:

        path = output_dir / filename

        if path.exists():
            try:
                path.unlink()
            except Exception as exc:
                warn(
                    f"Could not remove {path}: {exc}"
                )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Cached/resumable H2 robustness evaluation "
            "for YOLOv8, RT-DETR and RTMDet."
        )
    )

    parser.add_argument(
        "--test-images",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--test-labels",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--corruption-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--yolov8",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rtdetr",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rtmdet",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rtmdet-config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=False,
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete H2 cache and recompute everything.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images for debugging.",
    )

    args = parser.parse_args()

    # Default output if omitted.
    if args.output is None:

        script_path = Path(__file__).resolve()

        # .../hypothesis_tests/scripts/H2/script.py
        hypothesis_tests_dir = script_path.parents[2]

        args.output = (
            hypothesis_tests_dir
            / "results2"
            / "H2"
        )

    return args


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_input_paths(
    args: argparse.Namespace,
) -> None:

    paths = {
        "test-images": args.test_images,
        "test-labels": args.test_labels,
        "corruption-root": args.corruption_root,
        "yolov8": args.yolov8,
        "rtdetr": args.rtdetr,
        "rtmdet": args.rtmdet,
        "rtmdet-config": args.rtmdet_config,
    }

    banner("INPUT PATH VALIDATION")

    for name, path in paths.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} does not exist:\n{path}"
            )

        ok(f"{name}: {path}")

    if args.imgsz <= 0:
        raise ValueError(
            "--imgsz must be positive."
        )

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(
            "--conf must be between 0 and 1."
        )

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(
            "--iou must be between 0 and 1."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be positive."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    output_dir = Path(args.output)
    cache_dir = output_dir / "cache"

    ensure_dir(output_dir)
    ensure_dir(cache_dir)

    banner(
        "H2 ROBUSTNESS EVALUATION\n"
        "CACHED / RESUMABLE / IMAGE-ID SAFE VERSION"
    )

    print(
        f"Python executable:\n{sys.executable}"
    )

    print(
        f"Output directory:\n{output_dir}"
    )

    validate_input_paths(args)

    # -------------------------------------------------------------------------
    # FORCE
    # -------------------------------------------------------------------------

    if args.force:

        banner("FORCE MODE")

        handle_force(
            output_dir
        )

        ensure_dir(cache_dir)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    run_config = build_run_configuration(
        args
    )

    save_json(
        output_dir
        / "H2_run_configuration.json",
        run_config,
    )

    # -------------------------------------------------------------------------
    # Clean image map
    # -------------------------------------------------------------------------

    banner("CLEAN TEST IMAGE DISCOVERY")

    clean_map, clean_duplicates = build_image_map(
        args.test_images,
        "clean test images",
    )

    clean_count = len(clean_map)

    print(
        f"Clean test images: {clean_count}"
    )

    if args.limit is not None:

        print(
            f"Debug limit: {args.limit}"
        )

        if args.limit > clean_count:
            warn(
                "--limit is greater than available "
                "images; using all images."
            )

    # -------------------------------------------------------------------------
    # Corruption validation
    # -------------------------------------------------------------------------

    corruption_maps, validation_results = (
        validate_all_corruptions(
            clean_map,
            args.corruption_root,
        )
    )

    save_json(
        output_dir
        / "H2_image_id_validation.json",
        {
            "clean_image_count": clean_count,
            "results": validation_results,
        },
    )

    banner("IMAGE-ID VALIDATION PASSED")

    print(
        "All corruption datasets contain exactly "
        "the same normalized image IDs as the clean test set."
    )

    # -------------------------------------------------------------------------
    # Selected image IDs
    # -------------------------------------------------------------------------

    selected_clean_ids = sorted(
        clean_map.keys()
    )

    if args.limit is not None:
        selected_clean_ids = selected_clean_ids[
            :args.limit
        ]

    selected_clean_map = {
        image_id: clean_map[image_id]
        for image_id in selected_clean_ids
    }

    selected_corruption_maps = {}

    for corruption in CORRUPTIONS:

        selected_corruption_maps[corruption] = {
            image_id:
                corruption_maps[corruption][image_id]
            for image_id in selected_clean_ids
        }

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    models = load_models(
        args
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    banner("INFERENCE / CACHE")

    all_per_image_rows = []

    for model_name in MODELS:

        banner(
            f"MODEL: {model_name}"
        )

        # =====================================================================
        # CLEAN
        # =====================================================================

        clean_prediction_records = (
            run_or_resume_inference(
                model_name=model_name,
                condition="clean",
                image_map=selected_clean_map,
                models=models,
                args=args,
                cache_dir=cache_dir,
            )
        )

        clean_rows = evaluate_condition_model(
            model_name=model_name,
            condition="clean",
            image_map=selected_clean_map,
            prediction_records=clean_prediction_records,
            labels_dir=args.test_labels,
            iou_threshold=args.iou,
        )

        all_per_image_rows.extend(
            clean_rows
        )

        print(
            f"Clean evaluated rows: "
            f"{len(clean_rows)}"
        )

        # =====================================================================
        # CORRUPTIONS
        # =====================================================================

        for corruption in CORRUPTIONS:

            condition_map = (
                selected_corruption_maps[
                    corruption
                ]
            )

            prediction_records = (
                run_or_resume_inference(
                    model_name=model_name,
                    condition=corruption,
                    image_map=condition_map,
                    models=models,
                    args=args,
                    cache_dir=cache_dir,
                )
            )

            rows = evaluate_condition_model(
                model_name=model_name,
                condition=corruption,
                image_map=condition_map,
                prediction_records=prediction_records,
                labels_dir=args.test_labels,
                iou_threshold=args.iou,
            )

            all_per_image_rows.extend(
                rows
            )

            print(
                f"{corruption} evaluated rows: "
                f"{len(rows)}"
            )

    # -------------------------------------------------------------------------
    # Combined per-image dataframe
    # -------------------------------------------------------------------------

    banner("BUILDING PER-IMAGE METRICS")

    if not all_per_image_rows:

        raise RuntimeError(
            "No evaluation rows were produced."
        )

    per_image_df = pd.DataFrame(
        all_per_image_rows
    )

    per_image_df = per_image_df.sort_values(
        [
            "model",
            "condition",
            "image_id",
        ]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Save clean / corrupted
    # -------------------------------------------------------------------------

    clean_df = per_image_df[
        per_image_df["condition"] == "clean"
    ].copy()

    corrupted_df = per_image_df[
        per_image_df["condition"] != "clean"
    ].copy()

    clean_df.to_csv(
        output_dir
        / "clean_per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corrupted_df.to_csv(
        output_dir
        / "corrupted_per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_image_df.to_csv(
        output_dir
        / "per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ok(
        "Per-image metric CSV files saved."
    )

    # -------------------------------------------------------------------------
    # Degradation
    # -------------------------------------------------------------------------

    banner("BUILDING PAIRED DEGRADATION")

    degradation_df = (
        build_degradation_dataframe(
            per_image_df
        )
    )

    degradation_df = degradation_df.sort_values(
        [
            "model",
            "condition",
            "image_id",
        ]
    ).reset_index(drop=True)

    degradation_df.to_csv(
        output_dir
        / "per_image_degradation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ok(
        f"Degradation rows: "
        f"{len(degradation_df)}"
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    banner("H2 SUMMARY")

    summary_df = build_h2_summary(
        degradation_df
    )

    summary_df = summary_df.sort_values(
        [
            "model",
            "condition",
        ]
    ).reset_index(drop=True)

    summary_df.to_csv(
        output_dir
        / "H2_degradation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Condition descriptive comparison
    # -------------------------------------------------------------------------

    condition_wins_df = build_condition_wins(
        degradation_df
    )

    condition_wins_df.to_csv(
        output_dir
        / "H2_condition_wins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    corruption_counts = {
        corruption: len(
            selected_corruption_maps[
                corruption
            ]
        )
        for corruption in CORRUPTIONS
    }

    integrity = build_integrity_report(
        per_image_df=per_image_df,
        degradation_df=degradation_df,
        clean_count=clean_count,
        corruption_counts=corruption_counts,
        limit=args.limit,
    )

    save_json(
        output_dir
        / "H2_integrity_check.json",
        integrity,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    banner("H2 RESULTS")

    print()
    print(
        f"Clean images used       : "
        f"{len(selected_clean_ids)}"
    )

    print(
        f"Clean rows              : "
        f"{len(clean_df)}"
    )

    print(
        f"Corrupted rows          : "
        f"{len(corrupted_df)}"
    )

    print(
        f"Degradation rows        : "
        f"{len(degradation_df)}"
    )

    print()

    for model in MODELS:

        model_summary = summary_df[
            summary_df["model"] == model
        ]

        print(
            f"\n{'-' * 80}"
        )
        print(
            f"{model}"
        )
        print(
            f"{'-' * 80}"
        )

        for _, row in model_summary.iterrows():

            print(
                f"{str(row['condition']):20s} "
                f"Clean F1 = "
                f"{row['mean_clean_f1']:.6f} | "
                f"Corrupt F1 = "
                f"{row['mean_corrupted_f1']:.6f} | "
                f"Drop = "
                f"{row['mean_relative_drop_percent']:.4f}% | "
                f"Retention = "
                f"{row['mean_retention_percent']:.4f}%"
            )

    # -------------------------------------------------------------------------
    # Integrity final
    # -------------------------------------------------------------------------

    banner("FINAL INTEGRITY CHECK")

    print(
        json.dumps(
            integrity,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )

    if integrity["complete"]:

        print()
        ok(
            "H2 integrity check PASSED."
        )

        print(
            "All expected clean/corrupted paired rows "
            "and degradation rows are present."
        )

    else:

        print()
        fail(
            "H2 integrity check FAILED."
        )

        print(
            "The run produced incomplete results. "
            "Because the cache is resumable, rerun the same command "
            "without --force."
        )

        # Do not delete cache.
        # Exit with non-zero status.
        raise RuntimeError(
            "H2 integrity check failed. "
            "Incomplete inference/cache detected."
        )

    # -------------------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------------------

    banner("OUTPUT FILES")

    output_files = [
        "clean_per_image_metrics.csv",
        "corrupted_per_image_metrics.csv",
        "per_image_metrics.csv",
        "per_image_degradation.csv",
        "H2_degradation_summary.csv",
        "H2_condition_wins.csv",
        "H2_integrity_check.json",
        "H2_image_id_validation.json",
        "H2_run_configuration.json",
    ]

    for filename in output_files:

        path = output_dir / filename

        if path.exists():

            print(
                f"[OK] {path}"
            )

    print(
        f"\nCache directory:\n{cache_dir}"
    )

    banner(
        "H2 ROBUSTNESS EVALUATION COMPLETED SUCCESSFULLY"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        fail(
            "Execution interrupted by user."
        )

        print(
            "Existing cache files were preserved."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        traceback.print_exc()

        sys.exit(1)