from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# H3 COMPLETE PIPELINE
# DOMAIN-SHIFT ROBUSTNESS EVALUATION
# YOLOv8n vs RT-DETR-L vs RTMDet-Tiny
#
# Statistical extensions:
#   - paired per-image analysis
#   - bootstrap 95% CI
#   - Wilcoxon signed-rank
#   - paired effect size
#   - Friedman test
#   - Kendall's W
#   - pairwise Wilcoxon
#   - Holm multiple-comparison correction
#
# Output:
#   source_per_image_metrics.csv
#   target_per_image_metrics.csv
#   per_image_domain_shift.csv
#   domain_shift_summary.csv
#   domain_shift_statistics.csv
#   model_comparison_statistics.csv
#   bootstrap_results.csv
#   H3_integrity_check.json
#   H3_run_configuration.json
# ==============================================================================


# ==============================================================================
# MODELS
# ==============================================================================

MODELS = [
    "YOLOv8n",
    "RT-DETR-L",
    "RTMDet-Tiny",
]


MODEL_KEYS = {
    "YOLOv8n": "YOLOv8n",
    "RT-DETR-L": "RT-DETR-L",
    "RTMDet-Tiny": "RTMDet-Tiny",
}


EPS = 1e-12


# ==============================================================================
# IOU
# ==============================================================================

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

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    inter_area = (
        inter_w * inter_h
    )

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - inter_area
    )

    if union <= 0:
        return 0.0

    return inter_area / union


# ==============================================================================
# MATCHING
# ==============================================================================

def match_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int]:

    if (
        not predictions
        and
        not ground_truths
    ):
        return 0, 0, 0

    if not predictions:
        return (
            0,
            0,
            len(ground_truths),
        )

    if not ground_truths:
        return (
            0,
            len(predictions),
            0,
        )

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

        for gt_idx, gt in enumerate(
            ground_truths
        ):

            if gt_idx in matched_gt:
                continue

            if (
                int(pred["class_id"])
                !=
                int(gt["class_id"])
            ):
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
            and
            best_iou >= iou_threshold
        ):

            tp += 1
            matched_gt.add(best_gt_idx)

        else:

            fp += 1

    fn = (
        len(ground_truths)
        -
        len(matched_gt)
    )

    return tp, fp, fn


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:

    if tp + fp > 0:

        precision = (
            tp /
            (tp + fp)
        )

    else:

        precision = (
            1.0
            if tp + fn == 0
            else 0.0
        )

    if tp + fn > 0:

        recall = (
            tp /
            (tp + fn)
        )

    else:

        recall = (
            1.0
            if fp == 0
            else 0.0
        )

    denominator = (
        precision + recall
    )

    if denominator > 0:

        f1 = (
            2.0
            * precision
            * recall
            /
            denominator
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


# ==============================================================================
# IMAGE SIZE
# ==============================================================================

def get_image_size(
    image_path: Path,
) -> Tuple[int, int]:

    from PIL import Image

    with Image.open(image_path) as image:

        return image.size


# ==============================================================================
# YOLO LABEL LOADER
# ==============================================================================

def load_yolo_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:

    ground_truths = []

    if not label_path.exists():
        return ground_truths

    try:

        text = label_path.read_text(
            encoding="utf-8"
        )

    except Exception:

        return ground_truths

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except Exception:

            continue

        x1 = (
            xc - w / 2.0
        ) * image_width

        y1 = (
            yc - h / 2.0
        ) * image_height

        x2 = (
            xc + w / 2.0
        ) * image_width

        y2 = (
            yc + h / 2.0
        ) * image_height

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
# IMAGE VALIDATION
# ==============================================================================

def validate_image_set(
    image_dir: Path,
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

    images = sorted(
        [
            p
            for p in image_dir.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in extensions
            )
        ]
    )

    return images


# ==============================================================================
# ULTRALYTICS PREDICTION
# ==============================================================================

def predict_ultralytics(
    model,
    image_path: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> List[Dict[str, Any]]:

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

    xyxy = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    predictions = []

    for bbox, class_id, score in zip(
        xyxy,
        classes,
        scores,
    ):

        score = float(score)

        if score < conf:
            continue

        predictions.append(
            {
                "class_id":
                    int(class_id),

                "confidence":
                    score,

                "bbox":
                    [
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

    from mmdet.apis import inference_detector

    result = inference_detector(
        model,
        str(image_path),
    )

    predictions = []

    # --------------------------------------------------------------------------
    # MMDetection 3.x
    # --------------------------------------------------------------------------

    if hasattr(
        result,
        "pred_instances",
    ):

        pred_instances = (
            result.pred_instances
        )

        if not hasattr(
            pred_instances,
            "bboxes",
        ):
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
                    "class_id":
                        int(label),

                    "confidence":
                        score,

                    "bbox":
                        [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],
                }
            )

        return predictions

    # --------------------------------------------------------------------------
    # Compatibility fallback
    # --------------------------------------------------------------------------

    if isinstance(result, tuple):

        result = result[0]

    if isinstance(result, list):

        for class_id, class_result in enumerate(
            result
        ):

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

                x1, y1, x2, y2, score = (
                    row[:5]
                )

                score = float(score)

                if score < conf:
                    continue

                predictions.append(
                    {
                        "class_id":
                            int(class_id),

                        "confidence":
                            score,

                        "bbox":
                            [
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2),
                            ],
                    }
                )

    return predictions


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(
    model_name: str,
    model,
    image_paths: List[Path],
    labels_dir: Path,
    domain_name: str,
    imgsz: int,
    conf: float,
    iou_threshold: float,
    device: str,
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print(
        f"EVALUATING {model_name} | "
        f"{domain_name}"
    )
    print("=" * 80)

    rows = []

    total = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        if (
            index % 100 == 0
            or
            index == total
        ):

            print(
                f"[{index:5d}/{total}] "
                f"{image_path.name}"
            )

        try:

            width, height = (
                get_image_size(
                    image_path
                )
            )

            label_path = (
                labels_dir
                /
                f"{image_path.stem}.txt"
            )

            ground_truths = (
                load_yolo_ground_truth(
                    label_path,
                    width,
                    height,
                )
            )

            if model_name in (
                "YOLOv8n",
                "RT-DETR-L",
            ):

                predictions = (
                    predict_ultralytics(
                        model,
                        image_path,
                        imgsz,
                        conf,
                        device,
                    )
                )

            elif model_name == "RTMDet-Tiny":

                predictions = (
                    predict_rtmdet(
                        model,
                        image_path,
                        conf,
                    )
                )

            else:

                raise ValueError(
                    f"Unknown model: "
                    f"{model_name}"
                )

            tp, fp, fn = (
                match_predictions(
                    predictions,
                    ground_truths,
                    iou_threshold,
                )
            )

            precision, recall, f1 = (
                calculate_metrics(
                    tp,
                    fp,
                    fn,
                )
            )

            rows.append(
                {
                    "image_id":
                        image_path.stem,

                    "domain":
                        domain_name,

                    "model":
                        model_name,

                    "tp":
                        tp,

                    "fp":
                        fp,

                    "fn":
                        fn,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "num_gt":
                        len(ground_truths),

                    "num_predictions":
                        len(predictions),

                    "evaluation_status":
                        "success",

                    "error":
                        "",
                }
            )

        except Exception as exc:

            print(
                f"\nERROR: {image_path.name}"
            )

            print(
                f"Reason: {exc}"
            )

            rows.append(
                {
                    "image_id":
                        image_path.stem,

                    "domain":
                        domain_name,

                    "model":
                        model_name,

                    "tp":
                        np.nan,

                    "fp":
                        np.nan,

                    "fn":
                        np.nan,

                    "precision":
                        np.nan,

                    "recall":
                        np.nan,

                    "f1":
                        np.nan,

                    "num_gt":
                        np.nan,

                    "num_predictions":
                        np.nan,

                    "evaluation_status":
                        "failed",

                    "error":
                        str(exc),
                }
            )

    return rows


# ==============================================================================
# DEGRADATION
# ==============================================================================

def calculate_degradation(
    source_score,
    target_score,
) -> np.ndarray:

    source_score = np.asarray(
        source_score,
        dtype=float,
    )

    target_score = np.asarray(
        target_score,
        dtype=float,
    )

    denominator = np.abs(
        source_score
    )

    degradation = np.empty_like(
        source_score,
        dtype=float,
    )

    valid = denominator > EPS

    degradation[valid] = (
        (
            source_score[valid]
            -
            target_score[valid]
        )
        /
        denominator[valid]
    )

    degradation[~valid] = (
        source_score[~valid]
        -
        target_score[~valid]
    )

    return degradation


# ==============================================================================
# BOOTSTRAP CI
# ==============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return (
            float(values[0]),
            float(values[0]),
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    indices = rng.integers(
        0,
        n,
        size=(
            n_bootstrap,
            n,
        ),
    )

    samples = values[
        indices
    ]

    means = samples.mean(
        axis=1
    )

    alpha = (
        1.0 - confidence
    )

    low = np.percentile(
        means,
        100 * alpha / 2,
    )

    high = np.percentile(
        means,
        100 * (
            1 - alpha / 2
        ),
    )

    return (
        float(low),
        float(high),
    )


# ==============================================================================
# PAIRED EFFECT SIZE
# ==============================================================================

def paired_cohens_d(
    source: np.ndarray,
    target: np.ndarray,
) -> float:

    source = np.asarray(
        source,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    differences = (
        source - target
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 2:
        return np.nan

    std = np.std(
        differences,
        ddof=1,
    )

    if std <= EPS:
        return np.nan

    return float(
        np.mean(differences)
        /
        std
    )


# ==============================================================================
# HOLM CORRECTION
# ==============================================================================

def holm_correction(
    p_values: List[float],
) -> List[float]:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p),
        np.nan,
    )

    valid_indices = [
        i
        for i, value in enumerate(p)
        if np.isfinite(value)
    ]

    if not valid_indices:
        return adjusted.tolist()

    sorted_indices = sorted(
        valid_indices,
        key=lambda i: p[i],
    )

    m = len(
        sorted_indices
    )

    previous = 0.0

    for rank, idx in enumerate(
        sorted_indices
    ):

        value = (
            (m - rank)
            * p[idx]
        )

        value = max(
            value,
            previous,
        )

        value = min(
            value,
            1.0,
        )

        adjusted[idx] = value
        previous = value

    return adjusted.tolist()


# ==============================================================================
# PAIRED WILCOXON
# ==============================================================================

def wilcoxon_test(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:

    try:

        from scipy.stats import wilcoxon

    except ImportError:

        return (
            np.nan,
            np.nan,
        )

    source = np.asarray(
        source,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    valid = (
        np.isfinite(source)
        &
        np.isfinite(target)
    )

    source = source[valid]
    target = target[valid]

    if len(source) < 2:
        return np.nan, np.nan

    differences = (
        source - target
    )

    nonzero = (
        np.abs(differences)
        > EPS
    )

    if not np.any(nonzero):

        return (
            0.0,
            1.0,
        )

    try:

        result = wilcoxon(
            source,
            target,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        return (
            float(result.statistic),
            float(result.pvalue),
        )

    except Exception:

        return (
            np.nan,
            np.nan,
        )


# ==============================================================================
# BUILD PAIRED DATA
# ==============================================================================

def build_paired_domain_shift(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    condition_name: str,
    condition_type: str,
) -> pd.DataFrame:

    source_base = (
        source_df[
            [
                "image_id",
                "model",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .copy()
        .rename(
            columns={
                "precision":
                    "source_precision",

                "recall":
                    "source_recall",

                "f1":
                    "source_f1",
            }
        )
    )

    target_base = (
        target_df[
            [
                "image_id",
                "model",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .copy()
        .rename(
            columns={
                "precision":
                    "target_precision",

                "recall":
                    "target_recall",

                "f1":
                    "target_f1",
            }
        )
    )

    merged = (
        source_base.merge(
            target_base,
            on=[
                "image_id",
                "model",
            ],
            how="inner",
            validate="one_to_one",
        )
        .copy()
    )

    if merged.empty:

        raise ValueError(
            "No paired Source/Target "
            "image-model records found."
        )

    merged[
        "absolute_f1_drop"
    ] = (
        merged["source_f1"]
        -
        merged["target_f1"]
    )

    merged[
        "degradation"
    ] = calculate_degradation(
        merged["source_f1"].to_numpy(),
        merged["target_f1"].to_numpy(),
    )

    merged[
        "degradation_percent"
    ] = (
        merged["degradation"]
        * 100.0
    )

    merged[
        "retention"
    ] = np.where(
        np.abs(
            merged["source_f1"]
        ) > EPS,

        (
            merged["target_f1"]
            /
            merged["source_f1"]
        ),

        np.nan,
    )

    merged[
        "retention_percent"
    ] = (
        merged["retention"]
        * 100.0
    )

    merged[
        "condition"
    ] = condition_name

    merged[
        "condition_type"
    ] = condition_type

    return merged[
        [
            "condition",
            "condition_type",
            "image_id",
            "model",
            "source_precision",
            "target_precision",
            "source_recall",
            "target_recall",
            "source_f1",
            "target_f1",
            "absolute_f1_drop",
            "degradation",
            "degradation_percent",
            "retention",
            "retention_percent",
        ]
    ].copy()


# ==============================================================================
# SUMMARY
# ==============================================================================

def create_summary(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:

    rows = []

    for model in MODELS:

        source = (
            source_df[
                source_df["model"] == model
            ]["f1"]
            .dropna()
            .to_numpy()
        )

        target = (
            target_df[
                target_df["model"] == model
            ]["f1"]
            .dropna()
            .to_numpy()
        )

        paired = (
            paired_df[
                paired_df["model"] == model
            ]
            .dropna(
                subset=[
                    "source_f1",
                    "target_f1",
                    "degradation",
                ]
            )
            .copy()
        )

        degradation = (
            paired["degradation"]
            .to_numpy()
        )

        absolute_drop = (
            paired["absolute_f1_drop"]
            .to_numpy()
        )

        retention = (
            paired["retention"]
            .to_numpy()
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                degradation,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

        ret_ci_low, ret_ci_high = (
            bootstrap_mean_ci(
                retention,
                n_bootstrap=n_bootstrap,
                seed=seed + 100,
            )
        )

        source_target_ci_low, source_target_ci_high = (
            bootstrap_mean_ci(
                absolute_drop,
                n_bootstrap=n_bootstrap,
                seed=seed + 200,
            )
        )

        row = {

            "model":
                model,

            "n_source_images":
                len(source),

            "n_target_images":
                len(target),

            "n_paired_images":
                len(paired),

            "mean_source_f1":
                np.mean(source)
                if len(source)
                else np.nan,

            "median_source_f1":
                np.median(source)
                if len(source)
                else np.nan,

            "std_source_f1":
                np.std(
                    source,
                    ddof=1,
                )
                if len(source) > 1
                else np.nan,

            "mean_target_f1":
                np.mean(target)
                if len(target)
                else np.nan,

            "median_target_f1":
                np.median(target)
                if len(target)
                else np.nan,

            "std_target_f1":
                np.std(
                    target,
                    ddof=1,
                )
                if len(target) > 1
                else np.nan,

            "mean_absolute_f1_drop":
                np.mean(absolute_drop)
                if len(absolute_drop)
                else np.nan,

            "absolute_drop_ci95_low":
                source_target_ci_low,

            "absolute_drop_ci95_high":
                source_target_ci_high,

            "mean_degradation":
                np.mean(degradation)
                if len(degradation)
                else np.nan,

            "median_degradation":
                np.median(degradation)
                if len(degradation)
                else np.nan,

            "std_degradation":
                np.std(
                    degradation,
                    ddof=1,
                )
                if len(degradation) > 1
                else np.nan,

            "degradation_ci95_low":
                ci_low,

            "degradation_ci95_high":
                ci_high,

            "mean_degradation_percent":
                np.mean(degradation) * 100
                if len(degradation)
                else np.nan,

            "retention_mean":
                np.mean(retention)
                if len(retention)
                else np.nan,

            "retention_percent":
                np.mean(retention) * 100
                if len(retention)
                else np.nan,

            "retention_ci95_low":
                ret_ci_low,

            "retention_ci95_high":
                ret_ci_high,

            "degraded_images":
                int(
                    np.sum(
                        degradation > 0
                    )
                )
                if len(degradation)
                else 0,

            "improved_images":
                int(
                    np.sum(
                        degradation < 0
                    )
                )
                if len(degradation)
                else 0,

            "equal_images":
                int(
                    np.sum(
                        np.abs(
                            degradation
                        )
                        <= EPS
                    )
                )
                if len(degradation)
                else 0,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# H3 STATISTICS
# ==============================================================================

def create_h3_statistics(
    paired_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:

    rows = []

    raw_p_values = []

    temporary = []

    for model in MODELS:

        subset = (
            paired_df[
                paired_df["model"] == model
            ]
            .dropna(
                subset=[
                    "source_f1",
                    "target_f1",
                    "degradation",
                ]
            )
        )

        source = (
            subset["source_f1"]
            .to_numpy()
        )

        target = (
            subset["target_f1"]
            .to_numpy()
        )

        degradation = (
            subset["degradation"]
            .to_numpy()
        )

        statistic, p_value = (
            wilcoxon_test(
                source,
                target,
            )
        )

        effect = paired_cohens_d(
            source,
            target,
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                degradation,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

        temporary.append(
            {
                "comparison":
                    f"{model}: Source vs Target",

                "model":
                    model,

                "test":
                    "Wilcoxon signed-rank",

                "n":
                    len(source),

                "statistic":
                    statistic,

                "p_value":
                    p_value,

                "effect_size":
                    effect,

                "effect_size_type":
                    "paired Cohen's dz",

                "mean_degradation":
                    np.mean(degradation)
                    if len(degradation)
                    else np.nan,

                "ci95_low":
                    ci_low,

                "ci95_high":
                    ci_high,
            }
        )

        raw_p_values.append(
            p_value
        )

    adjusted = holm_correction(
        raw_p_values
    )

    for row, adjusted_p in zip(
        temporary,
        adjusted,
    ):

        row[
            "holm_adjusted_p"
        ] = adjusted_p

        row[
            "alpha"
        ] = 0.05

        row[
            "significant_after_holm"
        ] = (
            bool(
                np.isfinite(
                    adjusted_p
                )
                and
                adjusted_p < 0.05
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# FRIEDMAN TEST
# ==============================================================================

def create_model_comparison_statistics(
    paired_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    try:

        from scipy.stats import (
            friedmanchisquare,
            wilcoxon,
        )

    except ImportError:

        return pd.DataFrame(
            [
                {
                    "test":
                        "Friedman",

                    "error":
                        "scipy is not installed",
                }
            ]
        )

    pivot = (
        paired_df.pivot(
            index="image_id",
            columns="model",
            values="degradation",
        )
    )

    required = [
        model
        for model in MODELS
        if model in pivot.columns
    ]

    pivot = pivot[
        required
    ].dropna()

    if len(required) == 3 and len(pivot) >= 2:

        try:

            statistic, p_value = (
                friedmanchisquare(
                    pivot[
                        "YOLOv8n"
                    ],

                    pivot[
                        "RT-DETR-L"
                    ],

                    pivot[
                        "RTMDet-Tiny"
                    ],
                )
            )

            n = len(pivot)

            # Kendall's W for Friedman:
            # W = chi2 / [N * (k - 1)]
            k = 3

            kendall_w = (
                float(statistic)
                /
                (
                    n
                    * (k - 1)
                )
            )

            rows.append(
                {
                    "test":
                        "Friedman",

                    "comparison":
                        "YOLOv8n vs RT-DETR-L vs RTMDet-Tiny",

                    "n_paired_images":
                        n,

                    "statistic":
                        float(statistic),

                    "p_value":
                        float(p_value),

                    "effect_size":
                        kendall_w,

                    "effect_size_type":
                        "Kendall's W",

                    "holm_adjusted_p":
                        np.nan,

                    "significant":
                        bool(
                            p_value < 0.05
                        ),
                }
            )

        except Exception as exc:

            rows.append(
                {
                    "test":
                        "Friedman",

                    "comparison":
                        "YOLOv8n vs RT-DETR-L vs RTMDet-Tiny",

                    "n_paired_images":
                        len(pivot),

                    "statistic":
                        np.nan,

                    "p_value":
                        np.nan,

                    "effect_size":
                        np.nan,

                    "effect_size_type":
                        "Kendall's W",

                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------------------------
    # Pairwise model comparisons
    # --------------------------------------------------------------------------

    pair_definitions = [
        (
            "YOLOv8n",
            "RT-DETR-L",
        ),

        (
            "YOLOv8n",
            "RTMDet-Tiny",
        ),

        (
            "RT-DETR-L",
            "RTMDet-Tiny",
        ),
    ]

    pair_rows = []
    p_values = []

    for model_a, model_b in pair_definitions:

        if (
            model_a not in pivot.columns
            or
            model_b not in pivot.columns
        ):
            continue

        a = pivot[model_a]
        b = pivot[model_b]

        try:

            result = wilcoxon(
                a,
                b,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        except Exception:

            statistic = np.nan
            p_value = np.nan

        differences = (
            a.to_numpy()
            -
            b.to_numpy()
        )

        std = np.std(
            differences,
            ddof=1,
        )

        if std > EPS:

            effect = (
                np.mean(differences)
                /
                std
            )

        else:

            effect = np.nan

        pair_rows.append(
            {
                "test":
                    "Pairwise Wilcoxon",

                "comparison":
                    f"{model_a} vs {model_b}",

                "n_paired_images":
                    len(a),

                "statistic":
                    statistic,

                "p_value":
                    p_value,

                "effect_size":
                    effect,

                "effect_size_type":
                    "paired Cohen's dz",
            }
        )

        p_values.append(
            p_value
        )

    adjusted = holm_correction(
        p_values
    )

    for row, adjusted_p in zip(
        pair_rows,
        adjusted,
    ):

        row[
            "holm_adjusted_p"
        ] = adjusted_p

        row[
            "alpha"
        ] = 0.05

        row[
            "significant"
        ] = bool(
            np.isfinite(
                adjusted_p
            )
            and
            adjusted_p < 0.05
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# BOOTSTRAP RESULTS
# ==============================================================================

def create_bootstrap_results(
    paired_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:

    rows = []

    for model in MODELS:

        subset = (
            paired_df[
                paired_df["model"] == model
            ]
            .dropna(
                subset=[
                    "degradation",
                    "retention",
                ]
            )
        )

        degradation = (
            subset["degradation"]
            .to_numpy()
        )

        retention = (
            subset["retention"]
            .to_numpy()
        )

        d_low, d_high = (
            bootstrap_mean_ci(
                degradation,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

        r_low, r_high = (
            bootstrap_mean_ci(
                retention,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            )
        )

        rows.append(
            {
                "model":
                    model,

                "n":
                    len(degradation),

                "bootstrap_iterations":
                    n_bootstrap,

                "mean_degradation":
                    np.mean(degradation)
                    if len(degradation)
                    else np.nan,

                "ci95_low_degradation":
                    d_low,

                "ci95_high_degradation":
                    d_high,

                "mean_retention":
                    np.mean(retention)
                    if len(retention)
                    else np.nan,

                "ci95_low_retention":
                    r_low,

                "ci95_high_retention":
                    r_high,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# INTEGRITY CHECK
# ==============================================================================

def create_integrity_check(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    source_images: List[Path],
    target_images: List[Path],
    condition_name: str,
    condition_type: str,
) -> Dict[str, Any]:

    expected_source = (
        len(source_images)
        *
        len(MODELS)
    )

    expected_target = (
        len(target_images)
        *
        len(MODELS)
    )

    source_success = int(
        (
            source_df[
                "evaluation_status"
            ]
            ==
            "success"
        ).sum()
    )

    target_success = int(
        (
            target_df[
                "evaluation_status"
            ]
            ==
            "success"
        ).sum()
    )

    source_failed = (
        len(source_df)
        -
        source_success
    )

    target_failed = (
        len(target_df)
        -
        target_success
    )

    common_source_ids = set(
        source_df[
            "image_id"
        ]
        .dropna()
        .astype(str)
    )

    common_target_ids = set(
        target_df[
            "image_id"
        ]
        .dropna()
        .astype(str)
    )

    common_ids = (
        common_source_ids
        &
        common_target_ids
    )

    result = {

        "status":
            "PASS",

        "condition_name":
            condition_name,

        "condition_type":
            condition_type,

        "models":
            MODELS,

        "n_models":
            len(MODELS),

        "n_source_images":
            len(source_images),

        "n_target_images":
            len(target_images),

        "expected_source_rows":
            expected_source,

        "actual_source_rows":
            len(source_df),

        "expected_target_rows":
            expected_target,

        "actual_target_rows":
            len(target_df),

        "source_success_rows":
            source_success,

        "source_failed_rows":
            source_failed,

        "target_success_rows":
            target_success,

        "target_failed_rows":
            target_failed,

        "common_image_ids":
            len(common_ids),

        "paired_rows":
            len(paired_df),

        "nan_source_f1":
            int(
                source_df[
                    "f1"
                ].isna().sum()
            ),

        "nan_target_f1":
            int(
                target_df[
                    "f1"
                ].isna().sum()
            ),

        "nan_degradation":
            int(
                paired_df[
                    "degradation"
                ].isna().sum()
            ),

        "duplicate_source_image_model":
            int(
                source_df.duplicated(
                    subset=[
                        "image_id",
                        "model",
                    ]
                ).sum()
            ),

        "duplicate_target_image_model":
            int(
                target_df.duplicated(
                    subset=[
                        "image_id",
                        "model",
                    ]
                ).sum()
            ),

    }

    result[
        "complete"
    ] = bool(
        result[
            "actual_source_rows"
        ]
        ==
        expected_source

        and

        result[
            "actual_target_rows"
        ]
        ==
        expected_target

        and

        result[
            "source_failed_rows"
        ]
        ==
        0

        and

        result[
            "target_failed_rows"
        ]
        ==
        0

        and

        result[
            "nan_degradation"
        ]
        ==
        0

        and

        result[
            "duplicate_source_image_model"
        ]
        ==
        0

        and

        result[
            "duplicate_target_image_model"
        ]
        ==
        0
    )

    result[
        "status"
    ] = (
        "PASS"
        if result["complete"]
        else
        "FAIL"
    )

    return result


# ==============================================================================
# ENVIRONMENT INFO
# ==============================================================================

def collect_environment_info() -> Dict[str, Any]:

    info = {

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "python_executable":
            sys.executable,

    }

    try:

        import torch

        info[
            "torch_version"
        ] = torch.__version__

        info[
            "cuda_available"
        ] = bool(
            torch.cuda.is_available()
        )

        if torch.cuda.is_available():

            info[
                "cuda_version"
            ] = torch.version.cuda

            info[
                "gpu_name"
            ] = (
                torch.cuda.get_device_name(0)
            )

    except Exception as exc:

        info[
            "torch_error"
        ] = str(exc)

    try:

        import ultralytics

        info[
            "ultralytics_version"
        ] = ultralytics.__version__

    except Exception:
        pass

    try:

        import mmdet

        info[
            "mmdetection_version"
        ] = mmdet.__version__

    except Exception:
        pass

    return info


# ==============================================================================
# JSON
# ==============================================================================

def save_json(
    data: Dict[str, Any],
    path: Path,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# ==============================================================================
# FILE HASH
# ==============================================================================

def file_sha256(
    path: Path,
) -> str:

    sha256 = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Complete H3 Domain-Shift "
            "Robustness Evaluation"
        )
    )

    # --------------------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--source-images",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--source-labels",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target-images",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target-labels",
        required=True,
        type=Path,
    )

    # --------------------------------------------------------------------------
    # MODELS
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # EXPERIMENT META
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--condition-name",
        type=str,
        default="domain_shift",
    )

    parser.add_argument(
        "--condition-type",
        type=str,
        choices=[
            "real_domain_shift",
            "synthetic_transformation",
        ],
        default="real_domain_shift",
    )

    parser.add_argument(
        "--pairing-mode",
        choices=[
            "auto",
            "paired",
            "unpaired",
        ],
        default="auto",
    )

    # --------------------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------------------------

    if args.output is None:

        script_path = Path(
            __file__
        ).resolve()

        hypothesis_tests_dir = (
            script_path.parents[2]
        )

        output_dir = (
            hypothesis_tests_dir
            /
            "results"
            /
            "H3"
        )

    else:

        output_dir = (
            args.output
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================================
    # HEADER
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "H3 COMPLETE PIPELINE"
    )
    print(
        "DOMAIN-SHIFT ROBUSTNESS EVALUATION"
    )
    print(
        "YOLOv8n vs RT-DETR-L vs RTMDet-Tiny"
    )
    print("=" * 80)

    print()
    print(
        f"Condition      : "
        f"{args.condition_name}"
    )

    print(
        f"Condition type : "
        f"{args.condition_type}"
    )

    print(
        f"Image size     : "
        f"{args.imgsz}"
    )

    print(
        f"Confidence     : "
        f"{args.conf}"
    )

    print(
        f"IoU            : "
        f"{args.iou}"
    )

    print(
        f"Device         : "
        f"{args.device}"
    )

    print(
        f"Bootstrap      : "
        f"{args.bootstrap}"
    )

    print(
        f"Seed           : "
        f"{args.seed}"
    )

    print()
    print(
        f"Output         : "
        f"{output_dir}"
    )

    # ==========================================================================
    # VALIDATE PATHS
    # ==========================================================================

    required_paths = [

        args.source_images,
        args.source_labels,

        args.target_images,
        args.target_labels,

        args.yolov8,
        args.rtdetr,
        args.rtmdet,
        args.rtmdet_config,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path does not exist:\n"
                f"{path}"
            )

    # ==========================================================================
    # LOAD IMAGES
    # ==========================================================================

    source_images = (
        validate_image_set(
            args.source_images
        )
    )

    target_images = (
        validate_image_set(
            args.target_images
        )
    )

    if not source_images:

        raise RuntimeError(
            "No Source images found."
        )

    if not target_images:

        raise RuntimeError(
            "No Target images found."
        )

    print()
    print(
        f"Source images: "
        f"{len(source_images):,}"
    )

    print(
        f"Target images: "
        f"{len(target_images):,}"
    )

    # ==========================================================================
    # LOAD MODELS
    # ==========================================================================

    print()
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    from ultralytics import YOLO

    print()
    print("Loading YOLOv8n...")

    yolo_model = YOLO(
        str(args.yolov8)
    )

    print(
        "Loading RT-DETR-L..."
    )

    rtdetr_model = YOLO(
        str(args.rtdetr)
    )

    print(
        "Loading RTMDet-Tiny..."
    )

    from mmdet.apis import init_detector

    rtmdet_model = init_detector(
        str(args.rtmdet_config),
        str(args.rtmdet),
        device=args.device,
    )

    # ==========================================================================
    # SOURCE
    # ==========================================================================

    print()
    print("=" * 80)
    print("STEP 1/3 - SOURCE DOMAIN")
    print("=" * 80)

    source_rows = []

    source_rows.extend(
        evaluate_model(
            "YOLOv8n",
            yolo_model,
            source_images,
            args.source_labels,
            "source",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    source_rows.extend(
        evaluate_model(
            "RT-DETR-L",
            rtdetr_model,
            source_images,
            args.source_labels,
            "source",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    source_rows.extend(
        evaluate_model(
            "RTMDet-Tiny",
            rtmdet_model,
            source_images,
            args.source_labels,
            "source",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    source_df = pd.DataFrame(
        source_rows
    )

    source_output = (
        output_dir
        /
        "source_per_image_metrics.csv"
    )

    source_df.to_csv(
        source_output,
        index=False,
        encoding="utf-8-sig",
    )

    # ==========================================================================
    # TARGET
    # ==========================================================================

    print()
    print("=" * 80)
    print("STEP 2/3 - TARGET DOMAIN")
    print("=" * 80)

    target_rows = []

    target_rows.extend(
        evaluate_model(
            "YOLOv8n",
            yolo_model,
            target_images,
            args.target_labels,
            "target",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    target_rows.extend(
        evaluate_model(
            "RT-DETR-L",
            rtdetr_model,
            target_images,
            args.target_labels,
            "target",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    target_rows.extend(
        evaluate_model(
            "RTMDet-Tiny",
            rtmdet_model,
            target_images,
            args.target_labels,
            "target",
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )
    )

    target_df = pd.DataFrame(
        target_rows
    )

    target_output = (
        output_dir
        /
        "target_per_image_metrics.csv"
    )

    target_df.to_csv(
        target_output,
        index=False,
        encoding="utf-8-sig",
    )

    # ==========================================================================
    # VALID SUCCESS ROWS
    # ==========================================================================

    source_valid = (
        source_df[
            source_df[
                "evaluation_status"
            ]
            ==
            "success"
        ]
        .dropna(
            subset=[
                "image_id",
                "model",
                "f1",
            ]
        )
        .copy()
    )

    target_valid = (
        target_df[
            target_df[
                "evaluation_status"
            ]
            ==
            "success"
        ]
        .dropna(
            subset=[
                "image_id",
                "model",
                "f1",
            ]
        )
        .copy()
    )

    # ==========================================================================
    # PAIRING
    # ==========================================================================

    source_ids = set(
        source_valid[
            "image_id"
        ]
        .astype(str)
        .unique()
    )

    target_ids = set(
        target_valid[
            "image_id"
        ]
        .astype(str)
        .unique()
    )

    common_ids = (
        source_ids
        &
        target_ids
    )

    if args.pairing_mode == "paired":

        if not common_ids:

            raise RuntimeError(
                "Pairing requested but "
                "no common image IDs exist."
            )

        effective_pairing = "paired"

    elif args.pairing_mode == "unpaired":

        effective_pairing = "unpaired"

    else:

        effective_pairing = (
            "paired"
            if common_ids
            else
            "unpaired"
        )

    print()
    print("=" * 80)
    print("PAIRING ANALYSIS")
    print("=" * 80)

    print(
        f"Requested pairing : "
        f"{args.pairing_mode}"
    )

    print(
        f"Effective pairing : "
        f"{effective_pairing}"
    )

    print(
        f"Common image IDs  : "
        f"{len(common_ids):,}"
    )

    # ==========================================================================
    # PAIRED DATA
    # ==========================================================================

    paired_df = pd.DataFrame()

    if effective_pairing == "paired":

        print()
        print(
            "Building paired "
            "domain-shift data..."
        )

        paired_df = (
            build_paired_domain_shift(
                source_valid,
                target_valid,
                args.condition_name,
                args.condition_type,
            )
        )

        paired_output = (
            output_dir
            /
            "per_image_domain_shift.csv"
        )

        paired_df.to_csv(
            paired_output,
            index=False,
            encoding="utf-8-sig",
        )

    else:

        print()
        print(
            "UNPAIRED MODE:"
        )

        print(
            "Per-image degradation "
            "will not be calculated."
        )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    if effective_pairing == "paired":

        summary_df = create_summary(
            source_valid,
            target_valid,
            paired_df,
            args.bootstrap,
            args.seed,
        )

        summary_output = (
            output_dir
            /
            "domain_shift_summary.csv"
        )

        summary_df.to_csv(
            summary_output,
            index=False,
            encoding="utf-8-sig",
        )

        # ======================================================================
        # H3 STATISTICS
        # ======================================================================

        statistics_df = (
            create_h3_statistics(
                paired_df,
                args.bootstrap,
                args.seed,
            )
        )

        statistics_output = (
            output_dir
            /
            "domain_shift_statistics.csv"
        )

        statistics_df.to_csv(
            statistics_output,
            index=False,
            encoding="utf-8-sig",
        )

        # ======================================================================
        # MODEL COMPARISON
        # ======================================================================

        model_stats_df = (
            create_model_comparison_statistics(
                paired_df
            )
        )

        model_stats_output = (
            output_dir
            /
            "model_comparison_statistics.csv"
        )

        model_stats_df.to_csv(
            model_stats_output,
            index=False,
            encoding="utf-8-sig",
        )

        # ======================================================================
        # BOOTSTRAP
        # ======================================================================

        bootstrap_df = (
            create_bootstrap_results(
                paired_df,
                args.bootstrap,
                args.seed,
            )
        )

        bootstrap_output = (
            output_dir
            /
            "bootstrap_results.csv"
        )

        bootstrap_df.to_csv(
            bootstrap_output,
            index=False,
            encoding="utf-8-sig",
        )

    else:

        summary_df = pd.DataFrame()

        statistics_df = pd.DataFrame()

        model_stats_df = pd.DataFrame()

        bootstrap_df = pd.DataFrame()

    # ==========================================================================
    # INTEGRITY
    # ==========================================================================

    integrity = create_integrity_check(
        source_df,
        target_df,
        paired_df,
        source_images,
        target_images,
        args.condition_name,
        args.condition_type,
    )

    integrity_output = (
        output_dir
        /
        "H3_integrity_check.json"
    )

    save_json(
        integrity,
        integrity_output,
    )

    # ==========================================================================
    # RUN CONFIGURATION
    # ==========================================================================

    run_configuration = {

        "experiment":
            "H3 Domain-Shift Robustness",

        "condition_name":
            args.condition_name,

        "condition_type":
            args.condition_type,

        "model_names":
            MODELS,

        "source_images":
            str(
                args.source_images
            ),

        "source_labels":
            str(
                args.source_labels
            ),

        "target_images":
            str(
                args.target_images
            ),

        "target_labels":
            str(
                args.target_labels
            ),

        "checkpoints":
            {
                "YOLOv8n":
                    str(args.yolov8),

                "RT-DETR-L":
                    str(args.rtdetr),

                "RTMDet-Tiny":
                    str(args.rtmdet),
            },

        "rtmdet_config":
            str(args.rtmdet_config),

        "image_size":
            args.imgsz,

        "confidence_threshold":
            args.conf,

        "iou_threshold":
            args.iou,

        "device":
            args.device,

        "seed":
            args.seed,

        "bootstrap_iterations":
            args.bootstrap,

        "requested_pairing_mode":
            args.pairing_mode,

        "effective_pairing_mode":
            effective_pairing,

        "n_source_images":
            len(source_images),

        "n_target_images":
            len(target_images),

        "n_common_image_ids":
            len(common_ids),

        "statistical_tests":
            [
                "Wilcoxon signed-rank",
                "Friedman test",
                "Pairwise Wilcoxon",
                "Holm correction",
                "Paired Cohen's dz",
                "Bootstrap 95% CI",
                "Kendall's W",
            ],

        "degradation_formula":
            (
                "(source_f1 - target_f1) "
                "/ abs(source_f1)"
            ),

        "retention_formula":
            "target_f1 / source_f1",

        "evaluation_note":
            (
                "Test/evaluation labels are used "
                "only for evaluation and are not "
                "used for model training or "
                "checkpoint selection."
            ),

        "environment":
            collect_environment_info(),
    }

    config_output = (
        output_dir
        /
        "H3_run_configuration.json"
    )

    save_json(
        run_configuration,
        config_output,
    )

    # ==========================================================================
    # CONSOLE SUMMARY
    # ==========================================================================

    print()
    print("=" * 80)
    print("H3 DOMAIN-SHIFT SUMMARY")
    print("=" * 80)

    if not summary_df.empty:

        display_columns = [
            "model",
            "n_paired_images",
            "mean_source_f1",
            "mean_target_f1",
            "mean_absolute_f1_drop",
            "mean_degradation",
            "degradation_ci95_low",
            "degradation_ci95_high",
            "retention_percent",
            "degraded_images",
            "improved_images",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    # ==========================================================================
    # STATISTICS CONSOLE
    # ==========================================================================

    if not statistics_df.empty:

        print()
        print("=" * 80)
        print("H3 STATISTICAL TESTS")
        print("=" * 80)

        print(
            statistics_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    if not model_stats_df.empty:

        print()
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        print(
            model_stats_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    # ==========================================================================
    # INTEGRITY
    # ==========================================================================

    print()
    print("=" * 80)
    print("H3 INTEGRITY CHECK")
    print("=" * 80)

    for key, value in integrity.items():

        print(
            f"{key}: {value}"
        )

    # ==========================================================================
    # FINAL FILES
    # ==========================================================================

    print()
    print("=" * 80)
    print("H3 COMPLETE")
    print("=" * 80)

    print()
    print("Generated files:")

    for file_path in sorted(
        output_dir.iterdir()
    ):

        if file_path.is_file():

            print(
                f" - {file_path.name}"
            )

    print()

    if integrity["complete"]:

        print(
            "FINAL STATUS: "
            "H3 DATASET COMPLETE"
        )

    else:

        print(
            "FINAL STATUS: "
            "H3 DATASET INCOMPLETE"
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
            "H3 interrupted by user."
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

        traceback.print_exc()

        sys.exit(1)