# -*- coding: utf-8 -*-

"""
================================================================================
H1 - Controlled Baseline Comparison - RE-RUN VERSION
YOLOv8 vs RT-DETR vs RTMDet
================================================================================

هدف:
    اجرای مجدد H1 بدون حذف یا overwrite کردن نتایج قبلی.

خروجی اصلی:
    results2/H1/

ویژگی‌ها:
    1. اجرای inference برای YOLOv8
    2. اجرای inference برای RT-DETR
    3. اجرای inference برای RTMDet
    4. استفاده از TEST set مشترک
    5. IoU matching >= 0.50
    6. محاسبه TP / FP / FN
    7. محاسبه Precision / Recall / F1
    8. ذخیره تعداد matchهای موفق
    9. محاسبه Mean IoU برای matchهای موفق
    10. Friedman test
    11. Pairwise Wilcoxon
    12. Holm correction
    13. Kendall's W
    14. Rank-biserial correlation
    15. مقایسه نتیجه جدید با نتیجه قبلی در results/H1

نکته:
    هیچ آموزشی انجام نمی‌شود.
    فقط inference + evaluation + statistical analysis انجام می‌شود.

ساختار خروجی:

results2/
├── H1/
│   ├── per_image_metrics.csv
│   ├── H1_f1_paired_matrix.csv
│   ├── H1_pairwise_wilcoxon_holm.csv
│   └── H1_statistics_summary.json
│
└── comparison/
    └── H1_old_vs_new_summary.csv
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="H1 controlled baseline comparison - rerun"
    )

    parser.add_argument(
        "--test-images",
        type=Path,
        required=True,
        help="Path to TEST images directory"
    )

    parser.add_argument(
        "--test-labels",
        type=Path,
        required=True,
        help="Path to TEST YOLO labels directory"
    )

    parser.add_argument(
        "--yolov8",
        type=Path,
        required=True,
        help="YOLOv8 checkpoint (.pt)"
    )

    parser.add_argument(
        "--rtdetr",
        type=Path,
        required=True,
        help="RT-DETR checkpoint (.pt)"
    )

    parser.add_argument(
        "--rtmdet",
        type=Path,
        required=True,
        help="RTMDet checkpoint (.pth)"
    )

    parser.add_argument(
        "--rtmdet-config",
        type=Path,
        required=True,
        help="MMDetection RTMDet config (.py)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New output directory. Default: results2/H1"
    )

    parser.add_argument(
        "--previous-output",
        type=Path,
        default=None,
        help="Previous H1 directory. Default: results/H1"
    )

    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Comparison output directory. Default: results2/comparison"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold for TP matching"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu"
    )

    parser.add_argument(
        "--measure-time",
        action="store_true",
        help=(
            "Measure inference time and calculate FPS. "
            "Disabled by default."
        )
    )

    return parser.parse_args()


# =============================================================================
# PATH FUNCTIONS
# =============================================================================

def check_path(path: Path, name: str):

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


# =============================================================================
# IOU
# =============================================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0.0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1
    )

    intersection = (
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
        +
        area_b
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


# =============================================================================
# MATCHING
# =============================================================================

def match_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.50
):
    """
    Returns:

        tp
        fp
        fn
        matched_ious

    matched_ious contains IoU values for successful TP matches.
    """

    if (
        len(predictions) == 0
        and
        len(ground_truths) == 0
    ):

        return 0, 0, 0, []

    if len(predictions) == 0:

        return (
            0,
            0,
            len(ground_truths),
            []
        )

    if len(ground_truths) == 0:

        return (
            0,
            len(predictions),
            0,
            []
        )

    # Sort predictions by confidence
    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    matched_gt = set()

    matched_ious = []

    tp = 0
    fp = 0

    for pred in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(
            ground_truths
        ):

            if gt_index in matched_gt:
                continue

            # Class-aware matching
            if (
                pred["class_id"]
                !=
                gt["class_id"]
            ):
                continue

            iou = calculate_iou(
                pred["box"],
                gt["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and
            best_iou >= iou_threshold
        ):

            tp += 1

            matched_gt.add(
                best_gt_index
            )

            matched_ious.append(
                best_iou
            )

        else:

            fp += 1

    fn = (
        len(ground_truths)
        -
        len(matched_gt)
    )

    return (
        tp,
        fp,
        fn,
        matched_ious
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    tp,
    fp,
    fn
):

    if tp + fp > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = (
            1.0
            if (tp + fn) == 0
            else 0.0
        )

    if tp + fn > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = (
            1.0
            if fp == 0
            else 0.0
        )

    if precision + recall > 0:

        f1 = (
            2.0
            *
            precision
            *
            recall
            /
            (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1
    )


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_yolo_ground_truth(
    label_path: Path
):

    ground_truths = []

    if not label_path.exists():

        return ground_truths

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            warnings.warn(
                f"Invalid YOLO label line "
                f"in {label_path}: {line}"
            )

            continue

        try:

            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError:

            warnings.warn(
                f"Invalid numeric label "
                f"in {label_path}: {line}"
            )

            continue

        ground_truths.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

    return ground_truths


def convert_yolo_boxes(
    labels,
    image_width,
    image_height
):

    result = []

    for item in labels:

        xc = (
            item["x_center"]
            *
            image_width
        )

        yc = (
            item["y_center"]
            *
            image_height
        )

        w = (
            item["width"]
            *
            image_width
        )

        h = (
            item["height"]
            *
            image_height
        )

        x1 = xc - w / 2.0
        y1 = yc - h / 2.0

        x2 = xc + w / 2.0
        y2 = yc + h / 2.0

        result.append(
            {
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                "class_id":
                    item["class_id"]
            }
        )

    return result


# =============================================================================
# ULTRALYTICS
# =============================================================================

def load_ultralytics_model(
    weights
):

    try:

        from ultralytics import YOLO

    except ImportError:

        raise ImportError(
            "\nUltralytics is not installed "
            "in this environment."
        )

    return YOLO(
        str(weights)
    )


def predict_ultralytics(
    model,
    image_path,
    imgsz,
    conf,
    device,
    measure_time=False
):

    start_time = None

    if measure_time:

        start_time = time.perf_counter()

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False
    )

    elapsed = None

    if measure_time:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

    result = results[0]

    predictions = []

    if result.boxes is None:

        return predictions, elapsed

    boxes = result.boxes

    xyxy = (
        boxes.xyxy
        .cpu()
        .numpy()
    )

    scores = (
        boxes.conf
        .cpu()
        .numpy()
    )

    classes = (
        boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    for (
        box,
        score,
        class_id
    ) in zip(
        xyxy,
        scores,
        classes
    ):

        predictions.append(
            {
                "box":
                    box.tolist(),

                "class_id":
                    int(class_id),

                "confidence":
                    float(score)
            }
        )

    return (
        predictions,
        elapsed
    )


# =============================================================================
# RTMDET
# =============================================================================

def load_rtmdet(
    config_path,
    checkpoint_path,
    device
):

    try:

        from mmdet.apis import (
            init_detector
        )

    except ImportError:

        raise ImportError(
            "\nMMDetection is not available."
        )

    model = init_detector(
        str(config_path),
        str(checkpoint_path),
        device=device
    )

    return model


def predict_rtmdet(
    model,
    image_path,
    conf,
    measure_time=False
):

    try:

        from mmdet.apis import (
            inference_detector
        )

    except ImportError:

        raise ImportError(
            "MMDetection inference API "
            "is unavailable."
        )

    start_time = None

    if measure_time:

        start_time = time.perf_counter()

    result = inference_detector(
        model,
        str(image_path)
    )

    elapsed = None

    if measure_time:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

    predictions = []

    pred_instances = (
        result.pred_instances
    )

    boxes = (
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

    for (
        box,
        score,
        label
    ) in zip(
        boxes,
        scores,
        labels
    ):

        score = float(score)

        if score < conf:
            continue

        predictions.append(
            {
                "box":
                    box.tolist(),

                "class_id":
                    int(label),

                "confidence":
                    score
            }
        )

    return (
        predictions,
        elapsed
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model_name,
    model,
    image_paths,
    labels_dir,
    imgsz,
    conf,
    iou_threshold,
    device,
    is_rtmdet=False,
    measure_time=False
):

    rows = []

    total = len(
        image_paths
    )

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 80)

    for (
        index,
        image_path
    ) in enumerate(
        image_paths,
        start=1
    ):

        label_path = (
            labels_dir
            /
            f"{image_path.stem}.txt"
        )

        # ------------------------------------------------------------
        # IMAGE
        # ------------------------------------------------------------

        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            warnings.warn(
                f"Could not read image: "
                f"{image_path}"
            )

            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # ------------------------------------------------------------
        # GROUND TRUTH
        # ------------------------------------------------------------

        gt_raw = (
            load_yolo_ground_truth(
                label_path
            )
        )

        ground_truths = (
            convert_yolo_boxes(
                gt_raw,
                image_width,
                image_height
            )
        )

        # ------------------------------------------------------------
        # PREDICTIONS
        # ------------------------------------------------------------

        if is_rtmdet:

            (
                predictions,
                inference_time
            ) = predict_rtmdet(
                model,
                image_path,
                conf,
                measure_time
            )

        else:

            (
                predictions,
                inference_time
            ) = predict_ultralytics(
                model,
                image_path,
                imgsz,
                conf,
                device,
                measure_time
            )

        # ------------------------------------------------------------
        # MATCHING
        # ------------------------------------------------------------

        (
            tp,
            fp,
            fn,
            matched_ious
        ) = match_predictions(
            predictions,
            ground_truths,
            iou_threshold
        )

        # ------------------------------------------------------------
        # METRICS
        # ------------------------------------------------------------

        (
            precision,
            recall,
            f1
        ) = calculate_metrics(
            tp,
            fp,
            fn
        )

        # ------------------------------------------------------------
        # MATCH STATISTICS
        # ------------------------------------------------------------

        num_matched = len(
            matched_ious
        )

        if num_matched > 0:

            mean_iou = float(
                np.mean(
                    matched_ious
                )
            )

        else:

            mean_iou = 0.0

        # ------------------------------------------------------------
        # TIME / FPS
        # ------------------------------------------------------------

        if (
            measure_time
            and
            inference_time is not None
        ):

            inference_time_ms = (
                inference_time
                *
                1000.0
            )

            fps = (
                1.0
                /
                inference_time
                if inference_time > 0
                else np.nan
            )

        else:

            inference_time_ms = np.nan
            fps = np.nan

        # ------------------------------------------------------------
        # SAVE ROW
        # ------------------------------------------------------------

        rows.append(
            {
                "image_id":
                    image_path.stem,

                "model":
                    model_name,

                "tp":
                    int(tp),

                "fp":
                    int(fp),

                "fn":
                    int(fn),

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "f1":
                    float(f1),

                "num_gt":
                    int(len(ground_truths)),

                "num_predictions":
                    int(len(predictions)),

                "num_matched":
                    int(num_matched),

                "mean_iou":
                    float(mean_iou),

                "image_width":
                    int(image_width),

                "image_height":
                    int(image_height),

                "inference_time_ms":
                    inference_time_ms,

                "fps":
                    fps
            }
        )

        if (
            index % 100 == 0
            or
            index == total
        ):

            print(
                f"[{index:>5}/{total}] "
                f"{image_path.name}"
            )

    return rows


# =============================================================================
# STATISTICS
# =============================================================================

def rank_biserial_correlation(
    x,
    y
):

    diff = (
        np.asarray(x)
        -
        np.asarray(y)
    )

    diff = diff[
        diff != 0
    ]

    if len(diff) == 0:

        return 0.0

    ranks = rankdata(
        np.abs(diff),
        method="average"
    )

    positive = ranks[
        diff > 0
    ].sum()

    negative = ranks[
        diff < 0
    ].sum()

    denominator = (
        positive
        +
        negative
    )

    if denominator == 0:

        return 0.0

    return float(
        (
            positive
            -
            negative
        )
        /
        denominator
    )


def run_statistics(
    df
):

    models = [
        "YOLOv8",
        "RT-DETR",
        "RTMDet"
    ]

    pivot = df.pivot_table(
        index="image_id",
        columns="model",
        values="f1",
        aggfunc="mean"
    )

    pivot = pivot.dropna(
        subset=models
    )

    if len(pivot) < 3:

        raise ValueError(
            "Insufficient matched images "
            "for statistical analysis."
        )

    # -----------------------------------------------------------------
    # FRIEDMAN
    # -----------------------------------------------------------------

    friedman_stat, friedman_p = (
        friedmanchisquare(
            pivot["YOLOv8"],
            pivot["RT-DETR"],
            pivot["RTMDet"]
        )
    )

    n = len(pivot)
    k = len(models)

    kendalls_w = (
        friedman_stat
        /
        (
            n
            *
            (k - 1)
        )
    )

    # -----------------------------------------------------------------
    # PAIRWISE WILCOXON
    # -----------------------------------------------------------------

    pairs = [
        (
            "YOLOv8",
            "RT-DETR"
        ),
        (
            "YOLOv8",
            "RTMDet"
        ),
        (
            "RT-DETR",
            "RTMDet"
        )
    ]

    rows = []

    raw_p_values = []

    for (
        model_a,
        model_b
    ) in pairs:

        x = (
            pivot[model_a]
            .to_numpy()
        )

        y = (
            pivot[model_b]
            .to_numpy()
        )

        try:

            statistic, p_value = (
                wilcoxon(
                    x,
                    y,
                    zero_method="wilcox",
                    alternative="two-sided"
                )
            )

        except ValueError:

            statistic = np.nan
            p_value = 1.0

        effect = (
            rank_biserial_correlation(
                x,
                y
            )
        )

        raw_p_values.append(
            p_value
        )

        rows.append(
            {
                "comparison":
                    f"{model_a} vs {model_b}",

                "wilcoxon_statistic":
                    statistic,

                "raw_p":
                    p_value,

                "rank_biserial_r":
                    effect
            }
        )

    # -----------------------------------------------------------------
    # HOLM
    # -----------------------------------------------------------------

    (
        reject,
        corrected_p,
        _,
        _
    ) = multipletests(
        raw_p_values,
        method="holm",
        alpha=0.05
    )

    for (
        row,
        corrected,
        is_significant
    ) in zip(
        rows,
        corrected_p,
        reject
    ):

        row["holm_p"] = float(
            corrected
        )

        row[
            "significant_alpha_0.05"
        ] = bool(
            is_significant
        )

    pairwise_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    summary = {

        "analysis":
            "H1 Controlled Baseline Comparison",

        "primary_metric":
            "per_image_F1",

        "matching_iou_threshold":
            0.50,

        "n_matched_images":
            int(n),

        "number_of_models":
            int(k),

        "friedman_statistic":
            float(friedman_stat),

        "friedman_p_value":
            float(friedman_p),

        "kendalls_W":
            float(kendalls_w),

        "alpha":
            0.05,

        "pairwise_comparisons":
            rows
    }

    return (
        pivot,
        pairwise_df,
        summary
    )


# =============================================================================
# PREVIOUS RESULT COMPARISON
# =============================================================================

def safe_read_csv(
    path
):

    if not path.exists():

        return None

    try:

        return pd.read_csv(
            path
        )

    except Exception as exc:

        warnings.warn(
            f"Could not read {path}: {exc}"
        )

        return None


def compare_old_new(
    new_metrics_df,
    new_summary,
    previous_output,
    comparison_output
):

    comparison_output.mkdir(
        parents=True,
        exist_ok=True
    )

    previous_metrics_path = (
        previous_output
        /
        "per_image_metrics.csv"
    )

    previous_summary_path = (
        previous_output
        /
        "H1_statistics_summary.json"
    )

    old_df = safe_read_csv(
        previous_metrics_path
    )

    if old_df is None:

        print()
        print(
            "Previous per-image metrics "
            "not found."
        )

        return

    # ------------------------------------------------------------
    # MODEL LEVEL SUMMARY
    # ------------------------------------------------------------

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "num_gt",
        "num_predictions",
        "num_matched",
        "mean_iou"
    ]

    available_metrics = [
        col
        for col in metric_columns
        if (
            col in old_df.columns
            and
            col in new_metrics_df.columns
        )
    ]

    rows = []

    models = [
        "YOLOv8",
        "RT-DETR",
        "RTMDet"
    ]

    for model in models:

        old_model = old_df[
            old_df["model"] == model
        ]

        new_model = new_metrics_df[
            new_metrics_df["model"] == model
        ]

        if (
            old_model.empty
            or
            new_model.empty
        ):

            continue

        row = {
            "model": model,

            "old_n_images":
                int(len(old_model)),

            "new_n_images":
                int(len(new_model))
        }

        for metric in available_metrics:

            old_mean = float(
                old_model[metric]
                .mean()
            )

            new_mean = float(
                new_model[metric]
                .mean()
            )

            delta = (
                new_mean
                -
                old_mean
            )

            row[
                f"old_{metric}"
            ] = old_mean

            row[
                f"new_{metric}"
            ] = new_mean

            row[
                f"delta_{metric}"
            ] = delta

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_path = (
        comparison_output
        /
        "H1_old_vs_new_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # IMAGE-LEVEL COMPARISON
    # ------------------------------------------------------------

    merge_columns = [
        "image_id",
        "model"
    ]

    compare_columns = [
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "num_gt",
        "num_predictions",
        "num_matched",
        "mean_iou"
    ]

    old_available = [
        col
        for col in compare_columns
        if col in old_df.columns
    ]

    new_available = [
        col
        for col in compare_columns
        if col in new_metrics_df.columns
    ]

    old_subset = old_df[
        merge_columns
        +
        old_available
    ].copy()

    new_subset = new_metrics_df[
        merge_columns
        +
        new_available
    ].copy()

    old_subset = old_subset.rename(
        columns={
            col:
            f"old_{col}"
            for col in old_available
        }
    )

    new_subset = new_subset.rename(
        columns={
            col:
            f"new_{col}"
            for col in new_available
        }
    )

    image_comparison = pd.merge(
        old_subset,
        new_subset,
        on=merge_columns,
        how="outer"
    )

    for metric in compare_columns:

        old_col = (
            f"old_{metric}"
        )

        new_col = (
            f"new_{metric}"
        )

        if (
            old_col in image_comparison.columns
            and
            new_col in image_comparison.columns
        ):

            image_comparison[
                f"delta_{metric}"
            ] = (
                image_comparison[new_col]
                -
                image_comparison[old_col]
            )

    image_comparison_path = (
        comparison_output
        /
        "H1_old_vs_new_per_image.csv"
    )

    image_comparison.to_csv(
        image_comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # STATISTICS OLD VS NEW
    # ------------------------------------------------------------

    comparison_json = {
        "previous_output":
            str(previous_output),

        "new_output":
            str(
                comparison_output.parent
                /
                "H1"
            ),

        "new_statistics":
            new_summary
    }

    if previous_summary_path.exists():

        try:

            with open(
                previous_summary_path,
                "r",
                encoding="utf-8"
            ) as f:

                old_summary = json.load(
                    f
                )

            comparison_json[
                "previous_statistics"
            ] = old_summary

        except Exception as exc:

            comparison_json[
                "previous_statistics_error"
            ] = str(exc)

    comparison_json_path = (
        comparison_output
        /
        "H1_old_vs_new_statistics.json"
    )

    with open(
        comparison_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparison_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("OLD VS NEW COMPARISON")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Comparison files:"
    )

    print(
        summary_path
    )

    print(
        image_comparison_path
    )

    print(
        comparison_json_path
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print()
    print("=" * 80)
    print(
        "H1 CONTROLLED BASELINE COMPARISON - RE-RUN"
    )
    print("=" * 80)

    print(
        f"Test images : {args.test_images}"
    )

    print(
        f"Test labels : {args.test_labels}"
    )

    print(
        f"YOLOv8      : {args.yolov8}"
    )

    print(
        f"RT-DETR     : {args.rtdetr}"
    )

    print(
        f"RTMDet      : {args.rtmdet}"
    )

    print(
        f"RTMDet cfg  : {args.rtmdet_config}"
    )

    print(
        f"Image size  : {args.imgsz}"
    )

    print(
        f"Confidence  : {args.conf}"
    )

    print(
        f"IoU         : {args.iou}"
    )

    print(
        f"Device      : {args.device}"
    )

    print(
        f"Measure FPS : {args.measure_time}"
    )

    # =========================================================================
    # CHECK PATHS
    # =========================================================================

    check_path(
        args.test_images,
        "TEST images directory"
    )

    check_path(
        args.test_labels,
        "TEST labels directory"
    )

    check_path(
        args.yolov8,
        "YOLOv8 checkpoint"
    )

    check_path(
        args.rtdetr,
        "RT-DETR checkpoint"
    )

    check_path(
        args.rtmdet,
        "RTMDet checkpoint"
    )

    check_path(
        args.rtmdet_config,
        "RTMDet config"
    )

    # =========================================================================
    # OUTPUT DIRECTORIES
    # =========================================================================

    script_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    if args.output is None:

        output_dir = (
            script_root
            /
            "results2"
            /
            "H1"
        )

    else:

        output_dir = args.output

    if args.previous_output is None:

        previous_output = (
            script_root
            /
            "results"
            /
            "H1"
        )

    else:

        previous_output = (
            args.previous_output
        )

    if args.comparison_output is None:

        comparison_output = (
            script_root
            /
            "results2"
            /
            "comparison"
        )

    else:

        comparison_output = (
            args.comparison_output
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "NEW OUTPUT DIRECTORY:"
    )

    print(
        output_dir
    )

    print()
    print(
        "PREVIOUS OUTPUT DIRECTORY:"
    )

    print(
        previous_output
    )

    # =========================================================================
    # IMAGES
    # =========================================================================

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = sorted(
        [
            p
            for p in args.test_images.iterdir()
            if (
                p.suffix.lower()
                in valid_extensions
            )
        ]
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "No test images found."
        )

    print()
    print(
        f"Number of TEST images: "
        f"{len(image_paths)}"
    )

    # =========================================================================
    # LOAD YOLOv8
    # =========================================================================

    print()
    print(
        "Loading YOLOv8..."
    )

    yolov8_model = (
        load_ultralytics_model(
            args.yolov8
        )
    )

    # =========================================================================
    # LOAD RT-DETR
    # =========================================================================

    print(
        "Loading RT-DETR..."
    )

    rtdetr_model = (
        load_ultralytics_model(
            args.rtdetr
        )
    )

    # =========================================================================
    # LOAD RTMDET
    # =========================================================================

    print(
        "Loading RTMDet..."
    )

    rtmdet_model = load_rtmdet(
        args.rtmdet_config,
        args.rtmdet,
        args.device
    )

    # =========================================================================
    # EVALUATE YOLOv8
    # =========================================================================

    all_rows = []

    rows = evaluate_model(
        model_name="YOLOv8",
        model=yolov8_model,
        image_paths=image_paths,
        labels_dir=args.test_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        is_rtmdet=False,
        measure_time=args.measure_time
    )

    all_rows.extend(
        rows
    )

    # =========================================================================
    # EVALUATE RT-DETR
    # =========================================================================

    rows = evaluate_model(
        model_name="RT-DETR",
        model=rtdetr_model,
        image_paths=image_paths,
        labels_dir=args.test_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        is_rtmdet=False,
        measure_time=args.measure_time
    )

    all_rows.extend(
        rows
    )

    # =========================================================================
    # EVALUATE RTMDET
    # =========================================================================

    rows = evaluate_model(
        model_name="RTMDet",
        model=rtmdet_model,
        image_paths=image_paths,
        labels_dir=args.test_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        is_rtmdet=True,
        measure_time=args.measure_time
    )

    all_rows.extend(
        rows
    )

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    metrics_df = pd.DataFrame(
        all_rows
    )

    # Ensure deterministic column order
    columns = [
        "image_id",
        "model",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "num_gt",
        "num_predictions",
        "num_matched",
        "mean_iou",
        "image_width",
        "image_height",
        "inference_time_ms",
        "fps"
    ]

    columns = [
        col
        for col in columns
        if col in metrics_df.columns
    ]

    metrics_df = metrics_df[
        columns
    ]

    # =========================================================================
    # SAVE PER IMAGE
    # =========================================================================

    metrics_path = (
        output_dir
        /
        "per_image_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print(
        "PER-IMAGE METRICS SAVED"
    )
    print("=" * 80)

    print(
        metrics_path
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "RUNNING H1 STATISTICS"
    )
    print("=" * 80)

    (
        pivot,
        pairwise_df,
        summary
    ) = run_statistics(
        metrics_df
    )

    # =========================================================================
    # SAVE STATISTICS
    # =========================================================================

    pivot_path = (
        output_dir
        /
        "H1_f1_paired_matrix.csv"
    )

    pairwise_path = (
        output_dir
        /
        "H1_pairwise_wilcoxon_holm.csv"
    )

    summary_path = (
        output_dir
        /
        "H1_statistics_summary.json"
    )

    pivot.to_csv(
        pivot_path,
        encoding="utf-8-sig"
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =========================================================================
    # MODEL DESCRIPTIVE SUMMARY
    # =========================================================================

    descriptive_rows = []

    for model in [
        "YOLOv8",
        "RT-DETR",
        "RTMDet"
    ]:

        model_df = metrics_df[
            metrics_df["model"] == model
        ]

        if model_df.empty:
            continue

        descriptive_rows.append(
            {
                "model":
                    model,

                "n_images":
                    int(len(model_df)),

                "mean_precision":
                    float(
                        model_df[
                            "precision"
                        ].mean()
                    ),

                "mean_recall":
                    float(
                        model_df[
                            "recall"
                        ].mean()
                    ),

                "mean_f1":
                    float(
                        model_df[
                            "f1"
                        ].mean()
                    ),

                "median_f1":
                    float(
                        model_df[
                            "f1"
                        ].median()
                    ),

                "std_f1":
                    float(
                        model_df[
                            "f1"
                        ].std()
                    ),

                "mean_iou":
                    float(
                        model_df[
                            "mean_iou"
                        ].mean()
                    ),

                "mean_num_gt":
                    float(
                        model_df[
                            "num_gt"
                        ].mean()
                    ),

                "mean_num_predictions":
                    float(
                        model_df[
                            "num_predictions"
                        ].mean()
                    ),

                "mean_num_matched":
                    float(
                        model_df[
                            "num_matched"
                        ].mean()
                    )
            }
        )

    descriptive_df = pd.DataFrame(
        descriptive_rows
    )

    descriptive_path = (
        output_dir
        /
        "H1_descriptive_statistics.csv"
    )

    descriptive_df.to_csv(
        descriptive_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # OLD VS NEW
    # =========================================================================

    compare_old_new(
        new_metrics_df=metrics_df,
        new_summary=summary,
        previous_output=previous_output,
        comparison_output=comparison_output
    )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "H1 NEW RESULTS"
    )
    print("=" * 80)

    print(
        f"Matched images: "
        f"{summary['n_matched_images']}"
    )

    print(
        f"Friedman statistic: "
        f"{summary['friedman_statistic']:.6f}"
    )

    print(
        f"Friedman p-value: "
        f"{summary['friedman_p_value']:.6e}"
    )

    print(
        f"Kendall's W: "
        f"{summary['kendalls_W']:.6f}"
    )

    print()
    print(
        "PAIRWISE RESULTS:"
    )

    print(
        pairwise_df.to_string(
            index=False
        )
    )

    print()
    print(
        "DESCRIPTIVE STATISTICS:"
    )

    print(
        descriptive_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "FILES SAVED"
    )
    print("=" * 80)

    print(
        metrics_path
    )

    print(
        pivot_path
    )

    print(
        pairwise_path
    )

    print(
        summary_path
    )

    print(
        descriptive_path
    )

    print()
    print(
        "Comparison directory:"
    )

    print(
        comparison_output
    )

    print()
    print("=" * 80)
    print(
        "H1 RE-RUN COMPLETED"
    )
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "ERROR"
        )
        print("=" * 80)

        print(
            str(exc)
        )

        print()

        raise