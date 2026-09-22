# -*- coding: utf-8 -*-

"""
================================================================================
H1 - Controlled Baseline Comparison - FINAL CACHED VERSION
YOLOv8n vs RT-DETR-L vs RTMDet-Tiny
================================================================================

هدف:
    اجرای H1 بدون اجرای مجدد inference در صورت وجود prediction cache معتبر.

معیارهای اصلی:
    Precision
    Recall
    F1
    mAP@0.50
    mAP@0.50:0.95
    Mean IoU
    TP / FP / FN
    Number of Matches

آزمون‌های آماری:
    Friedman
    Pairwise Wilcoxon
    Holm correction
    Kendall's W
    Rank-biserial correlation

ویژگی مهم:
    اگر prediction cache معتبر وجود داشته باشد:
        inference مجدد انجام نمی‌شود.

    اگر فقط CSV قبلی وجود داشته باشد ولی prediction cache وجود نداشته باشد:
        inference یک بار انجام می‌شود و cache ساخته می‌شود.

خروجی:

results2/
├── H1/
│   ├── cache/
│   │   ├── YOLOv8_predictions.json
│   │   ├── RT-DETR_predictions.json
│   │   └── RTMDet_predictions.json
│   │
│   ├── per_image_metrics.csv
│   ├── H1_f1_paired_matrix.csv
│   ├── H1_pairwise_wilcoxon_holm.csv
│   ├── H1_statistics_summary.json
│   └── H1_descriptive_statistics.csv
│
└── comparison/
    ├── H1_old_vs_new_summary.csv
    ├── H1_old_vs_new_per_image.csv
    └── H1_old_vs_new_statistics.json

================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
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
        description="H1 controlled baseline comparison - cached final version"
    )

    parser.add_argument(
        "--test-images",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--test-labels",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--yolov8",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--rtdetr",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--rtmdet",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--rtmdet-config",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--previous-output",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    parser.add_argument(
        "--measure-time",
        action="store_true"
    )

    return parser.parse_args()


# =============================================================================
# PATH
# =============================================================================

def check_path(path: Path, name: str):

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


# =============================================================================
# FILE SIGNATURE
# =============================================================================

def file_signature(path: Path):

    stat = path.stat()

    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns
    }


def make_cache_signature(
    model_name,
    checkpoint,
    config_path,
    test_images,
    test_labels,
    imgsz,
    conf
):

    payload = {

        "model": model_name,

        "checkpoint":
            file_signature(checkpoint),

        "test_images":
            str(test_images.resolve()),

        "test_labels":
            str(test_labels.resolve()),

        "imgsz":
            imgsz,

        "conf":
            conf
    }

    if config_path is not None:

        payload["config"] = file_signature(
            config_path
        )

    raw = json.dumps(
        payload,
        sort_keys=True
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    payload["signature"] = digest

    return payload


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
                f"Invalid YOLO label line: "
                f"{label_path}: {line}"
            )

            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            warnings.warn(
                f"Invalid numeric label: "
                f"{label_path}: {line}"
            )

            continue

        ground_truths.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
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
                    int(item["class_id"])
            }
        )

    return result


# =============================================================================
# IMAGE LIST
# =============================================================================

def get_image_paths(test_images):

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return sorted(
        [
            p
            for p in test_images.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in valid_extensions
        ]
    )


# =============================================================================
# ULTRALYTICS
# =============================================================================

def load_ultralytics_model(weights):

    try:

        from ultralytics import YOLO

    except ImportError:

        raise ImportError(
            "Ultralytics is not installed."
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

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    scores = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    for box, score, class_id in zip(
        boxes,
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

    return predictions, elapsed


# =============================================================================
# RTMDET
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
            "MMDetection is not available."
        )

    return init_detector(
        str(config_path),
        str(checkpoint_path),
        device=device
    )


def predict_rtmdet(
    model,
    image_path,
    conf,
    measure_time=False
):

    try:

        from mmdet.apis import inference_detector

    except ImportError:

        raise ImportError(
            "MMDetection inference API unavailable."
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

    for box, score, label in zip(
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

    return predictions, elapsed


# =============================================================================
# CACHE
# =============================================================================

def cache_is_valid(
    cache_path,
    expected_signature
):

    if not cache_path.exists():

        return False

    try:

        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return (
            data.get("signature")
            ==
            expected_signature["signature"]
        )

    except Exception:

        return False


def save_prediction_cache(
    cache_path,
    signature,
    predictions_by_image
):

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {

        "signature":
            signature["signature"],

        "metadata":
            signature,

        "predictions":
            predictions_by_image
    }

    temporary_path = (
        cache_path.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    temporary_path.replace(
        cache_path
    )


def load_prediction_cache(
    cache_path
):

    with open(
        cache_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["predictions"]


# =============================================================================
# RUN INFERENCE WITH CACHE
# =============================================================================

def get_predictions_for_model(
    model_name,
    model,
    image_paths,
    labels_dir,
    cache_path,
    signature,
    imgsz,
    conf,
    device,
    is_rtmdet,
    measure_time
):

    # -------------------------------------------------------------------------
    # CACHE CHECK
    # -------------------------------------------------------------------------

    if cache_is_valid(
        cache_path,
        signature
    ):

        print()
        print(
            f"[CACHE HIT] {model_name}"
        )

        print(
            f"Loading predictions from:"
        )

        print(cache_path)

        predictions = (
            load_prediction_cache(
                cache_path
            )
        )

        return predictions

    # -------------------------------------------------------------------------
    # CACHE MISS
    # -------------------------------------------------------------------------

    print()
    print(
        f"[CACHE MISS] {model_name}"
    )

    print(
        "Inference will be executed once."
    )

    predictions_by_image = {}

    total = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        if is_rtmdet:

            predictions, elapsed = (
                predict_rtmdet(
                    model,
                    image_path,
                    conf,
                    measure_time
                )
            )

        else:

            predictions, elapsed = (
                predict_ultralytics(
                    model,
                    image_path,
                    imgsz,
                    conf,
                    device,
                    measure_time
                )
            )

        predictions_by_image[
            image_path.stem
        ] = {
            "predictions":
                predictions,

            "inference_time":
                elapsed
        }

        if (
            index % 100 == 0
            or
            index == total
        ):

            print(
                f"[{index:>5}/{total}] "
                f"{image_path.name}"
            )

    save_prediction_cache(
        cache_path,
        signature,
        predictions_by_image
    )

    print()
    print(
        f"[CACHE SAVED] {model_name}"
    )

    print(cache_path)

    return predictions_by_image


# =============================================================================
# MATCHING
# =============================================================================

def match_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.50
):

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

            if (
                pred["class_id"]
                !=
                gt["class_id"]
            ):
                continue

            current_iou = calculate_iou(
                pred["box"],
                gt["box"]
            )

            if current_iou > best_iou:

                best_iou = current_iou
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

def calculate_metrics(tp, fp, fn):

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

    if precision + recall > 0:

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (precision + recall)
        )

    else:

        f1 = 0.0

    return precision, recall, f1


# =============================================================================
# PER IMAGE EVALUATION FROM CACHE
# =============================================================================

def evaluate_from_cache(
    model_name,
    image_paths,
    labels_dir,
    prediction_cache,
    iou_threshold
):

    rows = []

    for image_path in image_paths:

        image_id = image_path.stem

        image_data = (
            prediction_cache.get(
                image_id,
                {}
            )
        )

        predictions = (
            image_data.get(
                "predictions",
                []
            )
        )

        label_path = (
            labels_dir
            /
            f"{image_id}.txt"
        )

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

        tp, fp, fn, matched_ious = (
            match_predictions(
                predictions,
                ground_truths,
                iou_threshold
            )
        )

        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn
            )
        )

        num_matched = len(
            matched_ious
        )

        mean_iou = (
            float(np.mean(matched_ious))
            if num_matched > 0
            else 0.0
        )

        inference_time = (
            image_data.get(
                "inference_time"
            )
        )

        if inference_time is not None:

            inference_time_ms = (
                inference_time * 1000.0
            )

            fps = (
                1.0 / inference_time
                if inference_time > 0
                else np.nan
            )

        else:

            inference_time_ms = np.nan
            fps = np.nan

        rows.append(
            {
                "image_id":
                    image_id,

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

    return rows


# =============================================================================
# AP / mAP
# =============================================================================

def compute_ap_101_point(
    recall,
    precision
):

    if len(recall) == 0:

        return 0.0

    recall = np.asarray(
        recall,
        dtype=float
    )

    precision = np.asarray(
        precision,
        dtype=float
    )

    ap = 0.0

    for threshold in np.linspace(
        0.0,
        1.0,
        101
    ):

        mask = (
            recall >= threshold
        )

        if np.any(mask):

            ap += np.max(
                precision[mask]
            )

    return float(
        ap / 101.0
    )


def compute_ap_for_class(
    image_paths,
    prediction_cache,
    labels_dir,
    class_id,
    iou_threshold
):

    detections = []

    gt_by_image = {}

    total_gt = 0

    for image_path in image_paths:

        image_id = image_path.stem

        label_path = (
            labels_dir
            /
            f"{image_id}.txt"
        )

        gt_raw = (
            load_yolo_ground_truth(
                label_path
            )
        )

        gt_for_class = []

        for item in gt_raw:

            if (
                item["class_id"]
                ==
                class_id
            ):

                gt_for_class.append(
                    item
                )

        gt_by_image[image_id] = (
            gt_for_class
        )

        total_gt += len(
            gt_for_class
        )

        image_data = (
            prediction_cache.get(
                image_id,
                {}
            )
        )

        for pred in image_data.get(
            "predictions",
            []
        ):

            if (
                pred["class_id"]
                ==
                class_id
            ):

                detections.append(
                    {
                        "image_id":
                            image_id,

                        "confidence":
                            float(
                                pred[
                                    "confidence"
                                ]
                            ),

                        "box":
                            pred["box"]
                    }
                )

    if total_gt == 0:

        return None

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    matched = {
        image_id:
        np.zeros(
            len(gt_boxes),
            dtype=bool
        )
        for image_id, gt_boxes
        in gt_by_image.items()
    }

    tp = np.zeros(
        len(detections)
    )

    fp = np.zeros(
        len(detections)
    )

    for index, detection in enumerate(
        detections
    ):

        image_id = detection[
            "image_id"
        ]

        gt_boxes = gt_by_image[
            image_id
        ]

        best_iou = 0.0
        best_index = None

        for gt_index, gt in enumerate(
            gt_boxes
        ):

            if matched[
                image_id
            ][gt_index]:

                continue

            gt_box = (
                convert_yolo_boxes(
                    [gt],
                    1.0,
                    1.0
                )[0]["box"]
            )

            # NOTE:
            # YOLO normalized GT needs image dimensions.
            # Reconstructing from 1x1 is incorrect for pixel predictions.
            # This is fixed below using image dimensions.
            del gt_box

        # Actual class matching is performed
        # using image dimensions below.

    # Rebuild using actual image dimensions
    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    matched = {}

    gt_pixel_by_image = {}

    for image_path in image_paths:

        image_id = image_path.stem

        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        label_path = (
            labels_dir
            /
            f"{image_id}.txt"
        )

        gt_raw = (
            load_yolo_ground_truth(
                label_path
            )
        )

        gt_pixel = (
            convert_yolo_boxes(
                [
                    x
                    for x in gt_raw
                    if x["class_id"] == class_id
                ],
                w,
                h
            )
        )

        gt_pixel_by_image[
            image_id
        ] = gt_pixel

        matched[
            image_id
        ] = np.zeros(
            len(gt_pixel),
            dtype=bool
        )

    tp[:] = 0
    fp[:] = 0

    for index, detection in enumerate(
        detections
    ):

        image_id = detection[
            "image_id"
        ]

        gt_boxes = gt_pixel_by_image.get(
            image_id,
            []
        )

        best_iou = 0.0
        best_index = None

        for gt_index, gt in enumerate(
            gt_boxes
        ):

            if matched[
                image_id
            ][gt_index]:

                continue

            current_iou = calculate_iou(
                detection["box"],
                gt["box"]
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_index = gt_index

        if (
            best_index is not None
            and
            best_iou >= iou_threshold
        ):

            tp[index] = 1

            matched[
                image_id
            ][best_index] = True

        else:

            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = (
        cumulative_tp
        /
        total_gt
    )

    precisions = (
        cumulative_tp
        /
        np.maximum(
            cumulative_tp
            +
            cumulative_fp,
            1e-12
        )
    )

    return compute_ap_101_point(
        recalls,
        precisions
    )


def calculate_map(
    image_paths,
    prediction_cache,
    labels_dir,
    iou_thresholds
):

    classes = set()

    for image_path in image_paths:

        label_path = (
            labels_dir
            /
            f"{image_path.stem}.txt"
        )

        gt = (
            load_yolo_ground_truth(
                label_path
            )
        )

        for item in gt:

            classes.add(
                int(item["class_id"])
            )

    if not classes:

        return {
            "mAP@0.50": 0.0,
            "mAP@0.50:0.95": 0.0,
            "per_class_ap": {}
        }

    results = {}

    for threshold in iou_thresholds:

        class_aps = []

        for class_id in sorted(
            classes
        ):

            ap = compute_ap_for_class(
                image_paths,
                prediction_cache,
                labels_dir,
                class_id,
                threshold
            )

            if ap is not None:

                class_aps.append(
                    ap
                )

                results[
                    f"{class_id}@{threshold:.2f}"
                ] = ap

        if class_aps:

            mean_ap = float(
                np.mean(class_aps)
            )

        else:

            mean_ap = 0.0

        if abs(
            threshold - 0.50
        ) < 1e-9:

            map50 = mean_ap

        if (
            abs(threshold - 0.50) < 1e-9
        ):

            pass

    map_50_95_values = []

    for threshold in iou_thresholds:

        values = []

        for class_id in sorted(
            classes
        ):

            key = (
                f"{class_id}@{threshold:.2f}"
            )

            if key in results:

                values.append(
                    results[key]
                )

        if values:

            map_50_95_values.append(
                np.mean(values)
            )

    map_50_95 = (
        float(
            np.mean(
                map_50_95_values
            )
        )
        if map_50_95_values
        else 0.0
    )

    return {
        "mAP@0.50":
            float(map50),

        "mAP@0.50:0.95":
            map_50_95,

        "per_class_ap":
            results
    }


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
        positive + negative
    )

    if denominator == 0:

        return 0.0

    return float(
        (
            positive - negative
        )
        /
        denominator
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
            "Insufficient paired images."
        )

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
            n * (k - 1)
        )
    )

    pairs = [
        ("YOLOv8", "RT-DETR"),
        ("YOLOv8", "RTMDet"),
        ("RT-DETR", "RTMDet")
    ]

    rows = []

    raw_p_values = []

    for model_a, model_b in pairs:

        x = pivot[
            model_a
        ].to_numpy()

        y = pivot[
            model_b
        ].to_numpy()

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

    reject, corrected_p, _, _ = (
        multipletests(
            raw_p_values,
            method="holm",
            alpha=0.05
        )
    )

    for row, corrected, significant in zip(
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
            significant
        )

    pairwise_df = pd.DataFrame(
        rows
    )

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

    return pivot, pairwise_df, summary


# =============================================================================
# OLD VS NEW
# =============================================================================

def safe_read_csv(path):

    if not path.exists():

        return None

    try:

        return pd.read_csv(path)

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

    old_df = safe_read_csv(
        previous_metrics_path
    )

    if old_df is None:

        print(
            "Previous metrics not found."
        )

        return

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
        "mean_iou",
        "mAP@0.50",
        "mAP@0.50:0.95"
    ]

    available = [
        x
        for x in metric_columns
        if (
            x in old_df.columns
            and
            x in new_metrics_df.columns
        )
    ]

    rows = []

    for model in [
        "YOLOv8",
        "RT-DETR",
        "RTMDet"
    ]:

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
            "model":
                model,

            "old_n_images":
                len(old_model),

            "new_n_images":
                len(new_model)
        }

        for metric in available:

            old_mean = float(
                old_model[metric].mean()
            )

            new_mean = float(
                new_model[metric].mean()
            )

            row[
                f"old_{metric}"
            ] = old_mean

            row[
                f"new_{metric}"
            ] = new_mean

            row[
                f"delta_{metric}"
            ] = (
                new_mean - old_mean
            )

        rows.append(row)

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

    print()
    print(
        "OLD VS NEW SUMMARY"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print()
    print("=" * 80)
    print(
        "H1 CONTROLLED BASELINE COMPARISON"
    )
    print(
        "CACHED + mAP VERSION"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECK INPUTS
    # -------------------------------------------------------------------------

    check_path(
        args.test_images,
        "TEST images"
    )

    check_path(
        args.test_labels,
        "TEST labels"
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

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    script_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    output_dir = (
        args.output
        if args.output is not None
        else
        script_root
        /
        "results2"
        /
        "H1"
    )

    previous_output = (
        args.previous_output
        if args.previous_output is not None
        else
        script_root
        /
        "results"
        /
        "H1"
    )

    comparison_output = (
        args.comparison_output
        if args.comparison_output is not None
        else
        script_root
        /
        "results2"
        /
        "comparison"
    )

    cache_dir = (
        output_dir
        /
        "cache"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # IMAGES
    # -------------------------------------------------------------------------

    image_paths = get_image_paths(
        args.test_images
    )

    if not image_paths:

        raise RuntimeError(
            "No test images found."
        )

    print()
    print(
        f"TEST images: {len(image_paths)}"
    )

    # -------------------------------------------------------------------------
    # MODEL DEFINITIONS
    # -------------------------------------------------------------------------

    model_specs = [
        {
            "name":
                "YOLOv8",

            "checkpoint":
                args.yolov8,

            "config":
                None,

            "is_rtmdet":
                False,

            "cache":
                cache_dir
                /
                "YOLOv8_predictions.json"
        },

        {
            "name":
                "RT-DETR",

            "checkpoint":
                args.rtdetr,

            "config":
                None,

            "is_rtmdet":
                False,

            "cache":
                cache_dir
                /
                "RT-DETR_predictions.json"
        },

        {
            "name":
                "RTMDet",

            "checkpoint":
                args.rtmdet,

            "config":
                args.rtmdet_config,

            "is_rtmdet":
                True,

            "cache":
                cache_dir
                /
                "RTMDet_predictions.json"
        }
    ]

    # -------------------------------------------------------------------------
    # LOAD MODELS ONLY WHEN CACHE IS MISSING
    # -------------------------------------------------------------------------

    loaded_models = {}

    prediction_caches = {}

    for spec in model_specs:

        signature = make_cache_signature(
            model_name=spec["name"],
            checkpoint=spec["checkpoint"],
            config_path=spec["config"],
            test_images=args.test_images,
            test_labels=args.test_labels,
            imgsz=args.imgsz,
            conf=args.conf
        )

        if cache_is_valid(
            spec["cache"],
            signature
        ):

            print()
            print(
                f"[CACHE VALID] "
                f"{spec['name']}"
            )

            prediction_caches[
                spec["name"]
            ] = load_prediction_cache(
                spec["cache"]
            )

            continue

        # ---------------------------------------------------------------------
        # Only load model when inference is actually needed
        # ---------------------------------------------------------------------

        print()
        print(
            f"Loading {spec['name']} "
            f"because cache is missing/invalid..."
        )

        if spec["is_rtmdet"]:

            model = load_rtmdet(
                spec["config"],
                spec["checkpoint"],
                args.device
            )

        else:

            model = load_ultralytics_model(
                spec["checkpoint"]
            )

        loaded_models[
            spec["name"]
        ] = model

        prediction_caches[
            spec["name"]
        ] = get_predictions_for_model(
            model_name=spec["name"],
            model=model,
            image_paths=image_paths,
            labels_dir=args.test_labels,
            cache_path=spec["cache"],
            signature=signature,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            is_rtmdet=spec["is_rtmdet"],
            measure_time=args.measure_time
        )

    # -------------------------------------------------------------------------
    # EVALUATE FROM CACHE
    # -------------------------------------------------------------------------

    all_rows = []

    for spec in model_specs:

        rows = evaluate_from_cache(
            model_name=spec["name"],
            image_paths=image_paths,
            labels_dir=args.test_labels,
            prediction_cache=prediction_caches[
                spec["name"]
            ],
            iou_threshold=args.iou
        )

        all_rows.extend(
            rows
        )

    metrics_df = pd.DataFrame(
        all_rows
    )

    # -------------------------------------------------------------------------
    # mAP
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "CALCULATING mAP"
    )
    print("=" * 80)

    iou_thresholds = np.round(
        np.arange(
            0.50,
            0.96,
            0.05
        ),
        2
    )

    map_results = {}

    for spec in model_specs:

        model_name = spec["name"]

        print()
        print(
            f"mAP -> {model_name}"
        )

        map_result = calculate_map(
            image_paths=image_paths,
            prediction_cache=prediction_caches[
                model_name
            ],
            labels_dir=args.test_labels,
            iou_thresholds=iou_thresholds
        )

        map_results[
            model_name
        ] = map_result

        print(
            f"mAP@0.50       = "
            f"{map_result['mAP@0.50']:.6f}"
        )

        print(
            f"mAP@0.50:0.95 = "
            f"{map_result['mAP@0.50:0.95']:.6f}"
        )

    # -------------------------------------------------------------------------
    # ADD MODEL-LEVEL mAP TO EVERY ROW
    # -------------------------------------------------------------------------

    metrics_df[
        "mAP@0.50"
    ] = metrics_df[
        "model"
    ].map(
        {
            model:
                result["mAP@0.50"]
            for model, result
            in map_results.items()
        }
    )

    metrics_df[
        "mAP@0.50:0.95"
    ] = metrics_df[
        "model"
    ].map(
        {
            model:
                result["mAP@0.50:0.95"]
            for model, result
            in map_results.items()
        }
    )

    # -------------------------------------------------------------------------
    # SAVE PER IMAGE
    # -------------------------------------------------------------------------

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
    print(
        "Saved:"
    )

    print(metrics_path)

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    pivot, pairwise_df, summary = (
        run_statistics(
            metrics_df
        )
    )

    # Add mAP to summary JSON
    summary[
        "mAP_results"
    ] = map_results

    # -------------------------------------------------------------------------
    # SAVE STATISTICS
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # DESCRIPTIVE SUMMARY
    # -------------------------------------------------------------------------

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

                "mAP@0.50":
                    float(
                        model_df[
                            "mAP@0.50"
                        ].iloc[0]
                    ),

                "mAP@0.50:0.95":
                    float(
                        model_df[
                            "mAP@0.50:0.95"
                        ].iloc[0]
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

    # -------------------------------------------------------------------------
    # OLD VS NEW
    # -------------------------------------------------------------------------

    compare_old_new(
        new_metrics_df=metrics_df,
        new_summary=summary,
        previous_output=previous_output,
        comparison_output=comparison_output
    )

    # -------------------------------------------------------------------------
    # CONSOLE
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "H1 FINAL RESULTS"
    )
    print("=" * 80)

    print(
        f"Paired images: "
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
        "MODEL RESULTS"
    )

    print(
        descriptive_df.to_string(
            index=False
        )
    )

    print()
    print(
        "PAIRWISE WILCOXON + HOLM"
    )

    print(
        pairwise_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "FILES"
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
        "CACHE:"
    )

    for spec in model_specs:

        print(
            spec["cache"]
        )

    print()
    print("=" * 80)
    print(
        "H1 COMPLETED"
    )
    print("=" * 80)


# =============================================================================
# ENTRY
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

        raise