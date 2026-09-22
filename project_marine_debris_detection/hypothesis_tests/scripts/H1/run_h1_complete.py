# -*- coding: utf-8 -*-

"""
================================================================================
H1 - Controlled Baseline Comparison
YOLOv8 vs RT-DETR vs RTMDet
================================================================================

هدف:
    آزمون H1 با استفاده از عملکرد per-image روی TEST set مشترک.

روش:
    1. اجرای inference برای هر سه مدل
    2. استخراج bounding box ها
    3. تطبیق prediction و ground-truth با IoU >= 0.50
    4. محاسبه TP / FP / FN
    5. محاسبه Precision / Recall / F1 برای هر تصویر
    6. آزمون Friedman
    7. Wilcoxon signed-rank pairwise
    8. Holm correction
    9. Kendall's W
    10. Rank-biserial correlation

نکته مهم:
    این فایل هیچ آموزشی انجام نمی‌دهد.
    فقط inference + statistical analysis انجام می‌شود.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
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
        description="H1: YOLOv8 vs RT-DETR vs RTMDet baseline comparison"
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
        help="Output directory"
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

    return parser.parse_args()


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def check_path(path: Path, name: str):

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


def calculate_iou(box_a, box_b):

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

    if union <= 0:
        return 0.0

    return intersection / union


def match_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.50
):
    """
    predictions:
        [
            {
                "box": [x1,y1,x2,y2],
                "class_id": int,
                "confidence": float
            }
        ]

    ground_truths:
        [
            {
                "box": [x1,y1,x2,y2],
                "class_id": int
            }
        ]
    """

    if len(predictions) == 0 and len(ground_truths) == 0:
        return 0, 0, 0

    if len(predictions) == 0:
        return 0, 0, len(ground_truths)

    if len(ground_truths) == 0:
        return 0, len(predictions), 0

    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truths):

            if gt_index in matched_gt:
                continue

            if pred["class_id"] != gt["class_id"]:
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
            and best_iou >= iou_threshold
        ):

            tp += 1
            matched_gt.add(best_gt_index)

        else:

            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if (tp + fn) == 0 else 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0 if fp == 0 else 0.0

    if precision + recall > 0:
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return precision, recall, f1


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_yolo_ground_truth(label_path: Path):

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
                f"Invalid YOLO label line in {label_path}: {line}"
            )
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

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

        xc = item["x_center"] * image_width
        yc = item["y_center"] * image_height

        w = item["width"] * image_width
        h = item["height"] * image_height

        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0

        result.append(
            {
                "box": [x1, y1, x2, y2],
                "class_id": item["class_id"],
            }
        )

    return result


# =============================================================================
# YOLOv8 / RT-DETR
# =============================================================================

def load_ultralytics_model(weights):

    try:

        from ultralytics import YOLO

    except ImportError:

        raise ImportError(
            "\nUltralytics is not installed in this environment.\n"
            "Install it in the same environment used for YOLO/RT-DETR."
        )

    return YOLO(str(weights))


def predict_ultralytics(
    model,
    image_path,
    imgsz,
    conf,
    device
):

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False
    )

    result = results[0]

    predictions = []

    if result.boxes is None:
        return predictions

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, score, class_id in zip(
        xyxy,
        scores,
        classes
    ):

        predictions.append(
            {
                "box": box.tolist(),
                "class_id": int(class_id),
                "confidence": float(score),
            }
        )

    return predictions


# =============================================================================
# RTMDet / MMDetection
# =============================================================================

def load_rtmdet(
    config_path,
    checkpoint_path,
    device
):

    try:

        from mmdet.apis import init_detector

    except ImportError:

        raise ImportError(
            "\nMMDetection is not available.\n"
            "Run this script inside the RTMDet training environment."
        )

    model = init_detector(
        str(config_path),
        str(checkpoint_path),
        device=device
    )

    return model


def predict_rtmdet(
    model,
    image_path
):

    try:

        from mmdet.apis import inference_detector

    except ImportError:

        raise ImportError(
            "MMDetection inference API is unavailable."
        )

    result = inference_detector(
        model,
        str(image_path)
    )

    predictions = []

    pred_instances = result.pred_instances

    boxes = pred_instances.bboxes.detach().cpu().numpy()
    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        predictions.append(
            {
                "box": box.tolist(),
                "class_id": int(label),
                "confidence": float(score),
            }
        )

    return predictions


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
    is_rtmdet=False
):

    rows = []

    total = len(image_paths)

    print()
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    for index, image_path in enumerate(image_paths, start=1):

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        # ------------------------------------------------------------
        # image dimensions
        # ------------------------------------------------------------

        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            warnings.warn(
                f"Could not read image: {image_path}"
            )

            continue

        image_height, image_width = image.shape[:2]

        # ------------------------------------------------------------
        # ground truth
        # ------------------------------------------------------------

        gt_raw = load_yolo_ground_truth(
            label_path
        )

        ground_truths = convert_yolo_boxes(
            gt_raw,
            image_width,
            image_height
        )

        # ------------------------------------------------------------
        # predictions
        # ------------------------------------------------------------

        if is_rtmdet:

            predictions = predict_rtmdet(
                model,
                image_path
            )

            # RTMDet score filtering
            predictions = [
                p for p in predictions
                if p["confidence"] >= conf
            ]

        else:

            predictions = predict_ultralytics(
                model,
                image_path,
                imgsz,
                conf,
                device
            )

        # ------------------------------------------------------------
        # matching
        # ------------------------------------------------------------

        tp, fp, fn = match_predictions(
            predictions,
            ground_truths,
            iou_threshold
        )

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn
        )

        rows.append(
            {
                "image_id": image_path.stem,
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

        if index % 100 == 0 or index == total:

            print(
                f"[{index:>5}/{total}] "
                f"{image_path.name}"
            )

    return rows


# =============================================================================
# STATISTICS
# =============================================================================

def rank_biserial_correlation(x, y):

    diff = np.asarray(x) - np.asarray(y)

    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(diff),
        method="average"
    )

    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()

    denominator = positive + negative

    if denominator == 0:
        return 0.0

    return float(
        (positive - negative) / denominator
    )


def run_statistics(df):

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
            "Insufficient matched images for statistical analysis."
        )

    # -----------------------------------------------------------------
    # Friedman
    # -----------------------------------------------------------------

    friedman_stat, friedman_p = friedmanchisquare(
        pivot["YOLOv8"],
        pivot["RT-DETR"],
        pivot["RTMDet"]
    )

    n = len(pivot)
    k = len(models)

    kendalls_w = (
        friedman_stat
        / (n * (k - 1))
    )

    # -----------------------------------------------------------------
    # Pairwise Wilcoxon
    # -----------------------------------------------------------------

    pairs = [
        ("YOLOv8", "RT-DETR"),
        ("YOLOv8", "RTMDet"),
        ("RT-DETR", "RTMDet"),
    ]

    rows = []

    raw_p_values = []

    for model_a, model_b in pairs:

        x = pivot[model_a].to_numpy()
        y = pivot[model_b].to_numpy()

        try:

            statistic, p_value = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided"
            )

        except ValueError:

            statistic = np.nan
            p_value = 1.0

        effect = rank_biserial_correlation(
            x,
            y
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
                    effect,
            }
        )

    # -----------------------------------------------------------------
    # Holm correction
    # -----------------------------------------------------------------

    reject, corrected_p, _, _ = multipletests(
        raw_p_values,
        method="holm",
        alpha=0.05
    )

    for row, corrected, is_significant in zip(
        rows,
        corrected_p,
        reject
    ):

        row["holm_p"] = float(corrected)

        row["significant_alpha_0.05"] = bool(
            is_significant
        )

    pairwise_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = {

        "primary_metric": "per_image_F1",

        "n_matched_images": int(n),

        "number_of_models": int(k),

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
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print()
    print("=" * 80)
    print("H1 CONTROLLED BASELINE COMPARISON")
    print("=" * 80)

    print(f"Test images : {args.test_images}")
    print(f"Test labels : {args.test_labels}")
    print(f"YOLOv8      : {args.yolov8}")
    print(f"RT-DETR     : {args.rtdetr}")
    print(f"RTMDet      : {args.rtmdet}")
    print(f"RTMDet cfg  : {args.rtmdet_config}")
    print(f"Image size  : {args.imgsz}")
    print(f"Confidence  : {args.conf}")
    print(f"IoU         : {args.iou}")
    print(f"Device      : {args.device}")

    # -----------------------------------------------------------------
    # Check paths
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    if args.output is None:

        output_dir = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "H1"
        )

    else:

        output_dir = args.output

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Output directory:")
    print(output_dir)

    # -----------------------------------------------------------------
    # Images
    # -----------------------------------------------------------------

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
            if p.suffix.lower() in valid_extensions
        ]
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "No test images found."
        )

    print()
    print(
        f"Number of TEST images: {len(image_paths)}"
    )

    # -----------------------------------------------------------------
    # Load YOLOv8
    # -----------------------------------------------------------------

    print()
    print("Loading YOLOv8...")

    yolov8_model = load_ultralytics_model(
        args.yolov8
    )

    # -----------------------------------------------------------------
    # Load RT-DETR
    # -----------------------------------------------------------------

    print("Loading RT-DETR...")

    rtdetr_model = load_ultralytics_model(
        args.rtdetr
    )

    # -----------------------------------------------------------------
    # Load RTMDet
    # -----------------------------------------------------------------

    print("Loading RTMDet...")

    rtmdet_model = load_rtmdet(
        args.rtmdet_config,
        args.rtmdet,
        args.device
    )

    # -----------------------------------------------------------------
    # Evaluate YOLOv8
    # -----------------------------------------------------------------

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
        is_rtmdet=False
    )

    all_rows.extend(rows)

    # -----------------------------------------------------------------
    # Evaluate RT-DETR
    # -----------------------------------------------------------------

    rows = evaluate_model(
        model_name="RT-DETR",
        model=rtdetr_model,
        image_paths=image_paths,
        labels_dir=args.test_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        is_rtmdet=False
    )

    all_rows.extend(rows)

    # -----------------------------------------------------------------
    # Evaluate RTMDet
    # -----------------------------------------------------------------

    rows = evaluate_model(
        model_name="RTMDet",
        model=rtmdet_model,
        image_paths=image_paths,
        labels_dir=args.test_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        is_rtmdet=True
    )

    all_rows.extend(rows)

    # -----------------------------------------------------------------
    # Save per-image metrics
    # -----------------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_rows
    )

    metrics_path = (
        output_dir
        / "per_image_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print("PER-IMAGE METRICS SAVED")
    print("=" * 80)

    print(metrics_path)

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING H1 STATISTICS")
    print("=" * 80)

    pivot, pairwise_df, summary = run_statistics(
        metrics_df
    )

    # -----------------------------------------------------------------
    # Save statistics
    # -----------------------------------------------------------------

    pivot_path = (
        output_dir
        / "H1_f1_paired_matrix.csv"
    )

    pairwise_path = (
        output_dir
        / "H1_pairwise_wilcoxon_holm.csv"
    )

    summary_path = (
        output_dir
        / "H1_statistics_summary.json"
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

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("H1 RESULTS")
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
    print(pairwise_df.to_string(index=False))

    print()
    print("Files saved:")
    print(metrics_path)
    print(pivot_path)
    print(pairwise_path)
    print(summary_path)

    print()
    print("=" * 80)
    print("H1 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nInterrupted by user.")
        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)
        print(str(exc))
        print()

        raise