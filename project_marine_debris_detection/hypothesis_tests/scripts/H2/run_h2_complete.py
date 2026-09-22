# -*- coding: utf-8 -*-

"""
================================================================================
H2 ROBUSTNESS / CORRUPTION ANALYSIS
================================================================================

هدف:
    ارزیابی robustness سه مدل YOLOv8، RT-DETR و RTMDet
    در برابر corruptionهای کنترل‌شده تصویری.

مدل‌ها:
    1. YOLOv8
    2. RT-DETR
    3. RTMDet

Corruptions:
    1. brightness
    2. contrast
    3. gaussian_noise
    4. gaussian_blur
    5. motion_blur
    6. jpeg_compression

روش:
    - ارزیابی روی test set تمیز
    - ارزیابی همان تصاویر با corruption
    - محاسبه F1 برای هر تصویر
    - محاسبه degradation نسبت به clean F1
    - paired analysis در سطح image

تعریف degradation:

    اگر clean_f1 > 0:

        degradation =
            (clean_f1 - corrupted_f1) / clean_f1

    بنابراین:
        degradation > 0  -> افت عملکرد
        degradation = 0  -> بدون افت
        degradation < 0  -> بهبود تحت corruption

برای clean_f1 = 0:
    degradation به صورت absolute drop تعریف می‌شود:

        degradation = clean_f1 - corrupted_f1

که در این حالت:
        degradation = -corrupted_f1

تمام ارزیابی‌های detection:
    IoU >= threshold
    class-aware matching
    confidence threshold = 0.25
    image size = 640

================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ==============================================================================
# PATH SETUP
# ==============================================================================

SCRIPT_PATH = Path(__file__).resolve()

# hypothesis_tests/
HYPOTHESIS_TESTS_DIR = SCRIPT_PATH.parents[2]

# marine_debris_detection/
PROJECT = HYPOTHESIS_TESTS_DIR.parent

RESULTS_DIR = (
    HYPOTHESIS_TESTS_DIR /
    "results" /
    "H2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================================
# MODELS
# ==============================================================================

MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]


# ==============================================================================
# CORRUPTIONS
# ==============================================================================

EXPECTED_CORRUPTIONS = [
    "brightness",
    "contrast",
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "jpeg_compression",
]


# ==============================================================================
# IOU
# ==============================================================================

def calculate_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:
    """
    Calculate IoU between two xyxy bounding boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# ==============================================================================
# BOX MATCHING
# ==============================================================================

def match_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """
    Class-aware greedy matching.

    Returns:
        TP, FP, FN
    """

    if not predictions and not ground_truths:
        return 0, 0, 0

    if not predictions:
        return 0, 0, len(ground_truths)

    if not ground_truths:
        return 0, len(predictions), 0

    predictions_sorted = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred in predictions_sorted:

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(ground_truths):

            if gt_idx in matched_gt:
                continue

            # Class-aware matching
            if int(pred["class_id"]) != int(gt["class_id"]):
                continue

            iou = calculate_iou(
                pred["bbox"],
                gt["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_gt.add(best_gt_idx)

        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall and F1.
    """

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if (tp + fn) == 0 else 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0 if fp == 0 else 0.0

    if precision + recall > 0:
        f1 = (
            2.0 *
            precision *
            recall /
            (precision + recall)
        )
    else:
        f1 = 0.0

    return precision, recall, f1


# ==============================================================================
# GROUND TRUTH LOADER
# ==============================================================================

def load_yolo_ground_truth(
    label_path: Path,
) -> List[Dict[str, Any]]:
    """
    Load YOLO-format labels.

    Format:
        class_id x_center y_center width height

    Coordinates are normalized.
    """

    ground_truths = []

    if not label_path.exists():
        return ground_truths

    try:
        with open(
            label_path,
            "r",
            encoding="utf-8",
        ) as f:

            lines = f.readlines()

    except Exception as exc:
        print(
            f"WARNING: Could not read label file: "
            f"{label_path}"
        )
        print(f"Reason: {exc}")
        return ground_truths

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            print(
                f"WARNING: Invalid label line "
                f"{label_path}:{line_number}"
            )
            continue

        try:
            class_id = int(float(parts[0]))

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            print(
                f"WARNING: Invalid numeric values "
                f"{label_path}:{line_number}"
            )
            continue

        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0

        ground_truths.append(
            {
                "class_id": class_id,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

    return ground_truths


# ==============================================================================
# YOLO NORMALIZED BOX -> ABSOLUTE XYXY
# ==============================================================================

def convert_yolo_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Convert YOLO normalized xywh boxes to absolute xyxy.
    """

    results = []

    if boxes is None:
        return results

    for box in boxes:

        xc, yc, w, h = box

        x1 = (xc - w / 2.0) * image_width
        y1 = (yc - h / 2.0) * image_height
        x2 = (xc + w / 2.0) * image_width
        y2 = (yc + h / 2.0) * image_height

        results.append(
            {
                "class_id": int(box[0]),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

    return results


# ==============================================================================
# YOLO / RT-DETR PREDICTION
# ==============================================================================

def predict_ultralytics(
    model,
    image_path: Path,
    image_width: int,
    image_height: int,
    imgsz: int,
    conf: float,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Inference for YOLOv8 / RT-DETR using Ultralytics.
    """

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()

    predictions = []

    for bbox, class_id, score in zip(
        xyxy,
        cls,
        scores,
    ):

        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
            }
        )

    return predictions


# ==============================================================================
# RTMDET PREDICTION
# ==============================================================================

def predict_rtmdet(
    model,
    image_path: Path,
    conf: float,
) -> List[Dict[str, Any]]:
    """
    Inference using MMDetection RTMDet.
    """

    from mmdet.apis import inference_detector

    result = inference_detector(
        model,
        str(image_path),
    )

    predictions = []

    # MMDetection 3.x DetDataSample
    if hasattr(result, "pred_instances"):

        pred_instances = result.pred_instances

        if not hasattr(pred_instances, "bboxes"):
            return predictions

        bboxes = (
            pred_instances.bboxes
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            pred_instances.scores
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            pred_instances.labels
            .detach()
            .cpu()
            .numpy()
        )

        for bbox, score, label in zip(
            bboxes,
            scores,
            labels,
        ):

            score = float(score)

            if score < conf:
                continue

            predictions.append(
                {
                    "class_id": int(label),
                    "confidence": score,
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                }
            )

        return predictions

    # Compatibility fallback
    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, list):

        for class_id, class_result in enumerate(result):

            if class_result is None:
                continue

            class_result = np.asarray(
                class_result
            )

            if class_result.size == 0:
                continue

            if class_result.ndim != 2:
                continue

            for row in class_result:

                if len(row) < 5:
                    continue

                x1, y1, x2, y2, score = row[:5]

                score = float(score)

                if score < conf:
                    continue

                predictions.append(
                    {
                        "class_id": int(class_id),
                        "confidence": score,
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                    }
                )

    return predictions


# ==============================================================================
# IMAGE SIZE
# ==============================================================================

def get_image_size(
    image_path: Path,
) -> Tuple[int, int]:

    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size

    return width, height


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(
    model_name: str,
    model,
    image_paths: List[Path],
    labels_dir: Path,
    condition: str,
    imgsz: int,
    conf: float,
    iou_threshold: float,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate one model on one image set.

    IMPORTANT:
        The column is always named `corruption`.
        For clean data:
            corruption = clean
        For corrupted data:
            corruption = brightness, contrast, ...
    """

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {model_name} | "
        f"CORRUPTION: {condition}"
    )
    print("=" * 80)

    rows = []

    total = len(image_paths)

    for idx, image_path in enumerate(
        image_paths,
        start=1,
    ):

        if idx % 100 == 0 or idx == total:
            print(
                f"[{idx:5d}/{total}] "
                f"{image_path.name}"
            )

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        ground_truths = (
            load_yolo_ground_truth(
                label_path
            )
        )

        try:

            if model_name in (
                "YOLOv8",
                "RT-DETR",
            ):

                width, height = get_image_size(
                    image_path
                )

                predictions = predict_ultralytics(
                    model=model,
                    image_path=image_path,
                    image_width=width,
                    image_height=height,
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                )

            elif model_name == "RTMDet":

                predictions = predict_rtmdet(
                    model=model,
                    image_path=image_path,
                    conf=conf,
                )

            else:
                raise ValueError(
                    f"Unknown model: {model_name}"
                )

            tp, fp, fn = match_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                iou_threshold=iou_threshold,
            )

            precision, recall, f1 = (
                calculate_metrics(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
            )

            rows.append(
                {
                    "image_id": image_path.stem,
                    "corruption": condition,
                    "model": model_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "num_gt": len(ground_truths),
                    "num_predictions": len(predictions),
                }
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing image: "
                f"{image_path}"
            )
            print(
                f"Model: {model_name}"
            )
            print(
                f"Corruption: {condition}"
            )
            print(
                f"Reason: {exc}"
            )

            # Keep an explicit failure row
            # so integrity checking can detect it.
            rows.append(
                {
                    "image_id": image_path.stem,
                    "corruption": condition,
                    "model": model_name,
                    "tp": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "num_gt": len(ground_truths),
                    "num_predictions": np.nan,
                }
            )

    return rows


# ==============================================================================
# VALIDATE IMAGE SET
# ==============================================================================

def validate_image_set(
    image_dir: Path,
    expected_count: int | None = None,
) -> List[Path]:

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{image_dir}"
        )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )

    if expected_count is not None:
        if len(image_paths) != expected_count:
            print()
            print(
                "WARNING: Unexpected number of images."
            )
            print(
                f"Expected: {expected_count}"
            )
            print(
                f"Found   : {len(image_paths)}"
            )

    return image_paths


# ==============================================================================
# CHECK UNIQUE IMAGE IDS
# ==============================================================================

def check_duplicate_ids(
    df: pd.DataFrame,
    name: str,
) -> None:

    duplicates = (
        df[
            df.duplicated(
                subset=[
                    "image_id",
                    "model",
                    "corruption",
                ],
                keep=False,
            )
        ]
    )

    if not duplicates.empty:

        print()
        print(
            f"WARNING: Duplicate records found "
            f"in {name}"
        )

        print(
            duplicates[
                [
                    "image_id",
                    "model",
                    "corruption",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


# ==============================================================================
# BUILD DEGRADATION DATAFRAME
# ==============================================================================

def build_degradation_dataframe(
    clean_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build paired clean-vs-corrupted dataframe.

    Pairing keys:
        image_id + model

    Output:
        image_id
        corruption
        model
        clean_f1
        corrupted_f1
        degradation
        clean_precision
        corrupted_precision
        clean_recall
        corrupted_recall
    """

    print()
    print("=" * 80)
    print("CALCULATING PER-IMAGE DEGRADATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------------------

    required_clean = {
        "image_id",
        "model",
        "corruption",
        "f1",
        "precision",
        "recall",
    }

    required_corrupted = required_clean.copy()

    missing_clean = (
        required_clean -
        set(clean_df.columns)
    )

    missing_corrupted = (
        required_corrupted -
        set(corrupted_df.columns)
    )

    if missing_clean:
        raise ValueError(
            "Clean dataframe is missing columns: "
            f"{sorted(missing_clean)}"
        )

    if missing_corrupted:
        raise ValueError(
            "Corrupted dataframe is missing columns: "
            f"{sorted(missing_corrupted)}"
        )

    # --------------------------------------------------------------------------
    # Remove accidental NaN evaluation rows
    # --------------------------------------------------------------------------

    clean_df = clean_df.copy()
    corrupted_df = corrupted_df.copy()

    # Clean must only contain clean rows.
    clean_df = clean_df[
        clean_df["corruption"].astype(str).str.lower()
        == "clean"
    ].copy()

    # Corrupted dataframe must not contain clean.
    corrupted_df = corrupted_df[
        corrupted_df["corruption"].astype(str).str.lower()
        != "clean"
    ].copy()

    # --------------------------------------------------------------------------
    # Clean dataframe
    # --------------------------------------------------------------------------

    clean_base = clean_df[
        [
            "image_id",
            "model",
            "f1",
            "precision",
            "recall",
        ]
    ].copy()

    clean_base = clean_base.rename(
        columns={
            "f1": "clean_f1",
            "precision": "clean_precision",
            "recall": "clean_recall",
        }
    )

    # --------------------------------------------------------------------------
    # Merge
    # --------------------------------------------------------------------------

    merged = corrupted_df[
        [
            "image_id",
            "corruption",
            "model",
            "f1",
            "precision",
            "recall",
        ]
    ].merge(
        clean_base,
        on=[
            "image_id",
            "model",
        ],
        how="left",
        validate="many_to_one",
    )

    merged = merged.rename(
        columns={
            "f1": "corrupted_f1",
            "precision": "corrupted_precision",
            "recall": "corrupted_recall",
        }
    )

    # --------------------------------------------------------------------------
    # Missing clean baseline
    # --------------------------------------------------------------------------

    missing_clean_rows = merged[
        merged["clean_f1"].isna()
    ]

    if not missing_clean_rows.empty:

        print()
        print(
            "WARNING: Missing clean baseline "
            "for some image/model pairs."
        )

        print(
            missing_clean_rows[
                [
                    "image_id",
                    "model",
                    "corruption",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------------------------
    # Degradation
    # --------------------------------------------------------------------------

    def calculate_degradation(row):

        clean_f1 = row["clean_f1"]
        corrupted_f1 = row["corrupted_f1"]

        if pd.isna(clean_f1) or pd.isna(corrupted_f1):
            return np.nan

        if clean_f1 > 0:
            return (
                (clean_f1 - corrupted_f1)
                / clean_f1
            )

        # Special case:
        # clean F1 = 0
        #
        # Use absolute change.
        return (
            clean_f1 -
            corrupted_f1
        )

    merged["degradation"] = merged.apply(
        calculate_degradation,
        axis=1,
    )

    # --------------------------------------------------------------------------
    # Additional absolute drop
    # --------------------------------------------------------------------------

    merged["absolute_f1_drop"] = (
        merged["clean_f1"]
        - merged["corrupted_f1"]
    )

    # --------------------------------------------------------------------------
    # Relative percentage degradation
    # --------------------------------------------------------------------------

    merged["degradation_percent"] = (
        merged["degradation"] * 100.0
    )

    # --------------------------------------------------------------------------
    # Column order
    # --------------------------------------------------------------------------

    preferred_columns = [
        "image_id",
        "corruption",
        "model",

        "clean_f1",
        "corrupted_f1",
        "degradation",
        "degradation_percent",
        "absolute_f1_drop",

        "clean_precision",
        "corrupted_precision",

        "clean_recall",
        "corrupted_recall",

        "tp",
        "fp",
        "fn",
        "num_gt",
        "num_predictions",
    ]

    # Add only columns that exist.
    final_columns = [
        col
        for col in preferred_columns
        if col in merged.columns
    ]

    remaining_columns = [
        col
        for col in merged.columns
        if col not in final_columns
    ]

    merged = merged[
        final_columns +
        remaining_columns
    ]

    return merged


# ==============================================================================
# INTEGRITY CHECK
# ==============================================================================

def integrity_check(
    clean_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    n_images: int,
    n_corruptions: int,
) -> Dict[str, Any]:
    """
    Validate expected H2 record counts.
    """

    expected_clean = (
        n_images *
        len(MODELS)
    )

    expected_corrupted = (
        n_images *
        n_corruptions *
        len(MODELS)
    )

    expected_degradation = expected_corrupted

    actual_clean = len(clean_df)
    actual_corrupted = len(corrupted_df)
    actual_degradation = len(degradation_df)

    result = {
        "n_images": n_images,
        "n_models": len(MODELS),
        "n_corruptions": n_corruptions,

        "expected_clean_rows": expected_clean,
        "actual_clean_rows": actual_clean,

        "expected_corrupted_rows": expected_corrupted,
        "actual_corrupted_rows": actual_corrupted,

        "expected_degradation_rows": (
            expected_degradation
        ),
        "actual_degradation_rows": (
            actual_degradation
        ),

        "clean_complete": (
            actual_clean ==
            expected_clean
        ),

        "corrupted_complete": (
            actual_corrupted ==
            expected_corrupted
        ),

        "degradation_complete": (
            actual_degradation ==
            expected_degradation
        ),
    }

    result["all_complete"] = (
        result["clean_complete"]
        and
        result["corrupted_complete"]
        and
        result["degradation_complete"]
    )

    return result


# ==============================================================================
# DESCRIPTIVE SUMMARY
# ==============================================================================

def create_h2_summary(
    degradation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive summary by corruption and model.
    """

    valid = degradation_df[
        degradation_df["degradation"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid
        .groupby(
            [
                "corruption",
                "model",
            ]
        )["degradation"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )

    summary["iqr"] = (
        summary["q3"] -
        summary["q1"]
    )

    return summary


# ==============================================================================
# SAVE JSON
# ==============================================================================

def save_json(
    data: Dict[str, Any],
    path: Path,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "H2 robustness/corruption analysis "
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
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    # ==========================================================================
    # HEADER
    # ==========================================================================

    print()
    print("=" * 80)
    print("H2 ROBUSTNESS / CORRUPTION ANALYSIS")
    print("=" * 80)

    print(
        f"Clean images       : {args.test_images}"
    )

    print(
        f"Test labels        : {args.test_labels}"
    )

    print(
        f"Corruption root    : {args.corruption_root}"
    )

    print(
        f"YOLOv8             : {args.yolov8}"
    )

    print(
        f"RT-DETR            : {args.rtdetr}"
    )

    print(
        f"RTMDet             : {args.rtmdet}"
    )

    print(
        f"RTMDet config      : {args.rtmdet_config}"
    )

    print(
        f"Image size         : {args.imgsz}"
    )

    print(
        f"Confidence         : {args.conf}"
    )

    print(
        f"IoU                : {args.iou}"
    )

    print(
        f"Device             : {args.device}"
    )

    print()
    print("Output directory:")
    print(RESULTS_DIR)

    # ==========================================================================
    # VALIDATE INPUTS
    # ==========================================================================

    required_paths = [
        args.test_images,
        args.test_labels,
        args.corruption_root,
        args.yolov8,
        args.rtdetr,
        args.rtmdet,
        args.rtmdet_config,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path does not exist:\n{path}"
            )

    # ==========================================================================
    # LOAD TEST IMAGES
    # ==========================================================================

    image_paths = validate_image_set(
        args.test_images
    )

    if not image_paths:
        raise RuntimeError(
            "No test images found."
        )

    print()
    print(
        f"Number of CLEAN test images: "
        f"{len(image_paths)}"
    )

    # ==========================================================================
    # DETERMINE CORRUPTION DIRECTORIES
    # ==========================================================================

    corruption_dirs = []

    for corruption in EXPECTED_CORRUPTIONS:

        directory = (
            args.corruption_root /
            corruption /
            "images"
        )

        if not directory.exists():

            print()
            print(
                "WARNING: Corruption directory "
                "does not exist:"
            )
            print(directory)

            continue

        corruption_images = validate_image_set(
            directory
        )

        if len(corruption_images) != len(
            image_paths
        ):

            print()
            print(
                f"WARNING: {corruption} contains "
                f"{len(corruption_images)} images; "
                f"expected {len(image_paths)}."
            )

        corruption_dirs.append(
            (
                corruption,
                directory,
                corruption_images,
            )
        )

    if not corruption_dirs:

        raise RuntimeError(
            "No valid corruption datasets found."
        )

    # ==========================================================================
    # IMPORT ULTRALYTICS
    # ==========================================================================

    print()
    print("Loading YOLOv8...")

    from ultralytics import YOLO

    yolo_model = YOLO(
        str(args.yolov8)
    )

    # ==========================================================================
    # LOAD RT-DETR
    # ==========================================================================

    print("Loading RT-DETR...")

    rtdetr_model = YOLO(
        str(args.rtdetr)
    )

    # ==========================================================================
    # LOAD RTMDET
    # ==========================================================================

    print("Loading RTMDet...")

    from mmdet.apis import init_detector

    rtmdet_model = init_detector(
        str(args.rtmdet_config),
        str(args.rtmdet),
        device=args.device,
    )

    # ==========================================================================
    # STEP 1: CLEAN
    # ==========================================================================

    print()
    print("=" * 80)
    print("STEP 1/2 - CLEAN BASELINE")
    print("=" * 80)

    clean_rows = []

    clean_rows.extend(
        evaluate_model(
            model_name="YOLOv8",
            model=yolo_model,
            image_paths=image_paths,
            labels_dir=args.test_labels,
            condition="clean",
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )
    )

    clean_rows.extend(
        evaluate_model(
            model_name="RT-DETR",
            model=rtdetr_model,
            image_paths=image_paths,
            labels_dir=args.test_labels,
            condition="clean",
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )
    )

    clean_rows.extend(
        evaluate_model(
            model_name="RTMDet",
            model=rtmdet_model,
            image_paths=image_paths,
            labels_dir=args.test_labels,
            condition="clean",
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )
    )

    clean_df = pd.DataFrame(
        clean_rows
    )

    clean_output = (
        RESULTS_DIR /
        "clean_per_image_metrics.csv"
    )

    clean_df.to_csv(
        clean_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Clean metrics saved:")
    print(clean_output)

    # ==========================================================================
    # STEP 2: CORRUPTED DATA
    # ==========================================================================

    print()
    print("=" * 80)
    print("STEP 2/2 - CORRUPTED DATASETS")
    print("=" * 80)

    corrupted_rows = []

    for (
        corruption,
        corruption_dir,
        corruption_images,
    ) in corruption_dirs:

        print()
        print("#" * 80)
        print(
            f"CORRUPTION: {corruption}"
        )
        print("#" * 80)

        print(
            f"Images available: "
            f"{len(corruption_images)}"
        )

        if len(corruption_images) == 0:
            print(
                "WARNING: No images found. "
                "Skipping."
            )
            continue

        # ----------------------------------------------------------------------
        # YOLOv8
        # ----------------------------------------------------------------------

        corrupted_rows.extend(
            evaluate_model(
                model_name="YOLOv8",
                model=yolo_model,
                image_paths=corruption_images,
                labels_dir=args.test_labels,
                condition=corruption,
                imgsz=args.imgsz,
                conf=args.conf,
                iou_threshold=args.iou,
                device=args.device,
            )
        )

        # ----------------------------------------------------------------------
        # RT-DETR
        # ----------------------------------------------------------------------

        corrupted_rows.extend(
            evaluate_model(
                model_name="RT-DETR",
                model=rtdetr_model,
                image_paths=corruption_images,
                labels_dir=args.test_labels,
                condition=corruption,
                imgsz=args.imgsz,
                conf=args.conf,
                iou_threshold=args.iou,
                device=args.device,
            )
        )

        # ----------------------------------------------------------------------
        # RTMDet
        # ----------------------------------------------------------------------

        corrupted_rows.extend(
            evaluate_model(
                model_name="RTMDet",
                model=rtmdet_model,
                image_paths=corruption_images,
                labels_dir=args.test_labels,
                condition=corruption,
                imgsz=args.imgsz,
                conf=args.conf,
                iou_threshold=args.iou,
                device=args.device,
            )
        )

    corrupted_df = pd.DataFrame(
        corrupted_rows
    )

    corrupted_output = (
        RESULTS_DIR /
        "corrupted_per_image_metrics.csv"
    )

    corrupted_df.to_csv(
        corrupted_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Corrupted metrics saved:")
    print(corrupted_output)

    # ==========================================================================
    # DUPLICATE CHECK
    # ==========================================================================

    check_duplicate_ids(
        clean_df,
        "clean dataframe",
    )

    check_duplicate_ids(
        corrupted_df,
        "corrupted dataframe",
    )

    # ==========================================================================
    # BUILD DEGRADATION
    # ==========================================================================

    degradation_df = build_degradation_dataframe(
        clean_df=clean_df,
        corrupted_df=corrupted_df,
    )

    # ==========================================================================
    # SAVE DEGRADATION
    # ==========================================================================

    degradation_output = (
        RESULTS_DIR /
        "per_image_degradation.csv"
    )

    degradation_df.to_csv(
        degradation_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Per-image degradation saved:"
    )
    print(degradation_output)

    # ==========================================================================
    # INTEGRITY CHECK
    # ==========================================================================

    integrity = integrity_check(
        clean_df=clean_df,
        corrupted_df=corrupted_df,
        degradation_df=degradation_df,
        n_images=len(image_paths),
        n_corruptions=len(corruption_dirs),
    )

    print()
    print("=" * 80)
    print("INTEGRITY CHECK")
    print("=" * 80)

    print(
        f"Number of images        : "
        f"{integrity['n_images']}"
    )

    print(
        f"Number of models        : "
        f"{integrity['n_models']}"
    )

    print(
        f"Number of corruptions   : "
        f"{integrity['n_corruptions']}"
    )

    print()

    print(
        f"Expected clean rows     : "
        f"{integrity['expected_clean_rows']}"
    )

    print(
        f"Actual clean rows       : "
        f"{integrity['actual_clean_rows']}"
    )

    print(
        f"Clean complete          : "
        f"{integrity['clean_complete']}"
    )

    print()

    print(
        f"Expected corrupted rows : "
        f"{integrity['expected_corrupted_rows']}"
    )

    print(
        f"Actual corrupted rows   : "
        f"{integrity['actual_corrupted_rows']}"
    )

    print(
        f"Corrupted complete      : "
        f"{integrity['corrupted_complete']}"
    )

    print()

    print(
        f"Expected degradation    : "
        f"{integrity['expected_degradation_rows']}"
    )

    print(
        f"Actual degradation      : "
        f"{integrity['actual_degradation_rows']}"
    )

    print(
        f"Degradation complete    : "
        f"{integrity['degradation_complete']}"
    )

    print()

    if integrity["all_complete"]:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print()
        print(
            "WARNING: H2 dataset is not complete."
        )

    # ==========================================================================
    # SAVE INTEGRITY JSON
    # ==========================================================================

    integrity_output = (
        RESULTS_DIR /
        "H2_integrity_check.json"
    )

    save_json(
        integrity,
        integrity_output,
    )

    print()
    print(
        "Integrity report saved:"
    )
    print(integrity_output)

    # ==========================================================================
    # DESCRIPTIVE SUMMARY
    # ==========================================================================

    summary_df = create_h2_summary(
        degradation_df
    )

    summary_output = (
        RESULTS_DIR /
        "H2_degradation_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "H2 degradation summary saved:"
    )
    print(summary_output)

    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================

    if not summary_df.empty:

        print()
        print("=" * 80)
        print("H2 DESCRIPTIVE SUMMARY")
        print("=" * 80)

        display_columns = [
            "corruption",
            "model",
            "n",
            "mean",
            "median",
            "q1",
            "q3",
            "iqr",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    # ==========================================================================
    # SAVE RUN CONFIGURATION
    # ==========================================================================

    run_config = {
        "test_images": str(
            args.test_images
        ),
        "test_labels": str(
            args.test_labels
        ),
        "corruption_root": str(
            args.corruption_root
        ),
        "yolov8": str(
            args.yolov8
        ),
        "rtdetr": str(
            args.rtdetr
        ),
        "rtmdet": str(
            args.rtmdet
        ),
        "rtmdet_config": str(
            args.rtmdet_config
        ),
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "iou": args.iou,
        "device": args.device,
        "models": MODELS,
        "corruptions": [
            item[0]
            for item in corruption_dirs
        ],
        "n_test_images": len(
            image_paths
        ),
    }

    config_output = (
        RESULTS_DIR /
        "H2_run_configuration.json"
    )

    save_json(
        run_config,
        config_output,
    )

    # ==========================================================================
    # FINAL
    # ==========================================================================

    print()
    print("=" * 80)
    print("H2 COMPLETE")
    print("=" * 80)

    print()
    print(
        "Generated files:"
    )

    print(
        f"1. {clean_output}"
    )

    print(
        f"2. {corrupted_output}"
    )

    print(
        f"3. {degradation_output}"
    )

    print(
        f"4. {summary_output}"
    )

    print(
        f"5. {integrity_output}"
    )

    print(
        f"6. {config_output}"
    )

    print()

    if integrity["all_complete"]:

        print(
            "FINAL STATUS: H2 DATASET COMPLETE"
        )

        print(
            "You can now run run_h2_statistics.py."
        )

    else:

        print(
            "FINAL STATUS: H2 DATASET INCOMPLETE"
        )

        print(
            "Do NOT run statistical analysis yet."
        )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "H2 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)
        print(
            str(exc)
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)