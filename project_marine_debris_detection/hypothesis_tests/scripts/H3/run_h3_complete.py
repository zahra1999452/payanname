

"""
================================================================================
H3 COMPLETE PIPELINE
DOMAIN-SHIFT ROBUSTNESS EVALUATION
YOLOv8 vs RT-DETR vs RTMDet
================================================================================

هدف:
    ارزیابی تفاوت میزان افت عملکرد سه مدل:

        1. YOLOv8
        2. RT-DETR
        3. RTMDet

    در هنگام انتقال از Source Domain به Target Domain.

فرضیه H3:
    میزان افت عملکرد مدل‌های YOLOv8، RT-DETR و RTMDet
    در شرایط Domain Shift یکسان نیست.

تعریف:

    degradation کمتر = robustness بیشتر

در حالت paired:

    degradation =
        (source_f1 - target_f1)
        / max(abs(source_f1), EPS)

تفسیر:

    degradation > 0
        افت عملکرد در Target Domain

    degradation = 0
        بدون تغییر عملکرد

    degradation < 0
        بهبود عملکرد در Target Domain


حالت‌های Pairing:

    auto
        تشخیص خودکار وجود image_id مشترک.

    paired
        Source و Target باید image_id مشترک داشته باشند.

    unpaired
        Source و Target مستقل هستند.
        در این حالت degradation per-image محاسبه نمی‌شود،
        زیرا pairing مصنوعی از نظر آماری صحیح نیست.


ورودی:

    --source-images
    --source-labels

    --target-images
    --target-labels

    --yolov8
    --rtdetr
    --rtmdet
    --rtmdet-config


خروجی:

    results/H3/

        source_per_image_metrics.csv
        target_per_image_metrics.csv

        per_image_domain_shift.csv
            فقط در حالت paired

        domain_shift_summary.csv

        H3_integrity_check.json
        H3_run_configuration.json


تمام ارزیابی‌های Detection:

    IoU >= threshold

    Class-aware matching

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


# ==============================================================================
# RESULTS
# ==============================================================================

RESULTS_DIR = (
    HYPOTHESIS_TESTS_DIR
    / "results"
    / "H3"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# CONSTANTS
# ==============================================================================

EPS = 1e-12


# ==============================================================================
# IOU
# ==============================================================================

def calculate_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:
    """
    Calculate IoU between two absolute xyxy bounding boxes.
    """

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
        inter_w
        * inter_h
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - inter_area
    )

    if union <= 0:
        return 0.0

    return (
        inter_area
        / union
    )


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
        TP
        FP
        FN
    """

    if (
        not predictions
        and not ground_truths
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
        key=lambda item:
            item["confidence"],
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

            # --------------------------------------------------------------
            # Class-aware matching
            # --------------------------------------------------------------

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

                best_gt_idx = (
                    gt_idx
                )

        if (
            best_gt_idx
            is not None
            and
            best_iou
            >= iou_threshold
        ):

            tp += 1

            matched_gt.add(
                best_gt_idx
            )

        else:

            fp += 1

    fn = (
        len(ground_truths)
        - len(matched_gt)
    )

    return (
        tp,
        fp,
        fn,
    )


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    """
    Calculate:

        Precision
        Recall
        F1
    """

    # --------------------------------------------------------------------------

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            / (
                tp
                + fp
            )
        )

    else:

        precision = (
            1.0
            if (
                tp + fn
            ) == 0
            else 0.0
        )

    # --------------------------------------------------------------------------

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            / (
                tp
                + fn
            )
        )

    else:

        recall = (
            1.0
            if fp == 0
            else 0.0
        )

    # --------------------------------------------------------------------------

    denominator = (
        precision
        + recall
    )

    if denominator > 0:

        f1 = (
            2.0
            * precision
            * recall
            / denominator
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


# ==============================================================================
# GROUND TRUTH LOADER
# ==============================================================================

def load_yolo_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Load YOLO labels and convert normalized xywh
    to absolute xyxy coordinates.
    """

    ground_truths = []

    if not label_path.exists():

        return ground_truths

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8",
        ) as file:

            lines = (
                file.readlines()
            )

    except Exception as exc:

        print()

        print(
            "WARNING: Could not read "
            "label file:"
        )

        print(
            label_path
        )

        print(
            f"Reason: {exc}"
        )

        return ground_truths

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:

            print(
                "WARNING: Invalid label line "
                f"{label_path}:{line_number}"
            )

            continue

        try:

            class_id = int(
                float(
                    parts[0]
                )
            )

            xc = float(
                parts[1]
            )

            yc = float(
                parts[2]
            )

            w = float(
                parts[3]
            )

            h = float(
                parts[4]
            )

        except ValueError:

            print(
                "WARNING: Invalid numeric "
                f"values {label_path}:{line_number}"
            )

            continue

        x1 = (
            xc
            - w / 2.0
        ) * image_width

        y1 = (
            yc
            - h / 2.0
        ) * image_height

        x2 = (
            xc
            + w / 2.0
        ) * image_width

        y2 = (
            yc
            + h / 2.0
        ) * image_height

        ground_truths.append(
            {
                "class_id":
                    class_id,

                "bbox":
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
            }
        )

    return ground_truths


# ==============================================================================
# IMAGE SIZE
# ==============================================================================

def get_image_size(
    image_path: Path,
) -> Tuple[int, int]:

    from PIL import Image

    with Image.open(
        image_path
    ) as image:

        width, height = (
            image.size
        )

    return (
        width,
        height,
    )


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
    """
    YOLOv8 / RT-DETR inference.
    """

    results = model.predict(
        source=str(
            image_path
        ),
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

    for (
        bbox,
        class_id,
        score,
    ) in zip(
        xyxy,
        classes,
        scores,
    ):

        predictions.append(
            {
                "class_id":
                    int(class_id),

                "confidence":
                    float(score),

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
    """
    RTMDet inference using MMDetection.
    """

    from mmdet.apis import (
        inference_detector
    )

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

        for (
            bbox,
            score,
            label,
        ) in zip(
            bboxes,
            scores,
            labels,
        ):

            score = float(
                score
            )

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

    if isinstance(
        result,
        tuple,
    ):

        result = (
            result[0]
        )

    if isinstance(
        result,
        list,
    ):

        for (
            class_id,
            class_result,
        ) in enumerate(
            result
        ):

            if class_result is None:
                continue

            class_result = (
                np.asarray(
                    class_result
                )
            )

            if class_result.size == 0:
                continue

            if class_result.ndim != 2:
                continue

            for row in class_result:

                if len(row) < 5:
                    continue

                (
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                ) = row[:5]

                score = float(
                    score
                )

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
# IMAGE SET VALIDATION
# ==============================================================================

def validate_image_set(
    image_dir: Path,
) -> List[Path]:

    if not image_dir.exists():

        raise FileNotFoundError(
            "Image directory does not "
            f"exist:\n{image_dir}"
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
            path
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in extensions
            )
        ]
    )

    return image_paths


# ==============================================================================
# EVALUATE MODEL
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
    """
    Evaluate one model on one domain.
    """

    print()

    print(
        "=" * 80
    )

    print(
        f"EVALUATING MODEL: "
        f"{model_name}"
    )

    print(
        f"DOMAIN: "
        f"{domain_name}"
    )

    print(
        "=" * 80
    )

    rows = []

    total = len(
        image_paths
    )

    for (
        index,
        image_path,
    ) in enumerate(
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
                / f"{image_path.stem}.txt"
            )

            ground_truths = (
                load_yolo_ground_truth(
                    label_path=
                        label_path,

                    image_width=
                        width,

                    image_height=
                        height,
                )
            )

            # --------------------------------------------------------------

            if model_name in (
                "YOLOv8",
                "RT-DETR",
            ):

                predictions = (
                    predict_ultralytics(
                        model=model,

                        image_path=
                            image_path,

                        imgsz=
                            imgsz,

                        conf=
                            conf,

                        device=
                            device,
                    )
                )

            elif (
                model_name
                == "RTMDet"
            ):

                predictions = (
                    predict_rtmdet(
                        model=model,

                        image_path=
                            image_path,

                        conf=
                            conf,
                    )
                )

            else:

                raise ValueError(
                    f"Unknown model: "
                    f"{model_name}"
                )

            # --------------------------------------------------------------

            tp, fp, fn = (
                match_predictions(
                    predictions=
                        predictions,

                    ground_truths=
                        ground_truths,

                    iou_threshold=
                        iou_threshold,
                )
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
                        len(
                            ground_truths
                        ),

                    "num_predictions":
                        len(
                            predictions
                        ),
                }
            )

        except Exception as exc:

            print()

            print(
                "ERROR processing image:"
            )

            print(
                image_path
            )

            print(
                f"Model: {model_name}"
            )

            print(
                f"Domain: {domain_name}"
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
                }
            )

    return rows


# ==============================================================================
# CHECK DUPLICATES
# ==============================================================================

def check_duplicate_ids(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    duplicates = (
        dataframe[
            dataframe.duplicated(
                subset=[
                    "image_id",
                    "model",
                ],
                keep=False,
            )
        ]
    )

    if not duplicates.empty:

        print()

        print(
            f"WARNING: Duplicate records "
            f"found in {name}"
        )

        print(
            duplicates[
                [
                    "image_id",
                    "model",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )


# ==============================================================================
# CALCULATE DEGRADATION
# ==============================================================================

def calculate_degradation(
    source_score,
    target_score,
) -> np.ndarray:
    """
    Relative performance degradation.

    degradation =
        (source - target)
        / abs(source)

    For source approximately zero:

        degradation =
            source - target
    """

    source_score = (
        np.asarray(
            source_score,
            dtype=float,
        )
    )

    target_score = (
        np.asarray(
            target_score,
            dtype=float,
        )
    )

    denominator = (
        np.abs(
            source_score
        )
    )

    degradation = (
        np.empty_like(
            source_score,
            dtype=float,
        )
    )

    valid = (
        denominator > EPS
    )

    degradation[
        valid
    ] = (

        (
            source_score[valid]
            - target_score[valid]
        )

        /

        denominator[valid]

    )

    invalid = (
        ~valid
    )

    degradation[
        invalid
    ] = (

        source_score[invalid]
        - target_score[invalid]

    )

    return degradation


# ==============================================================================
# BUILD PAIRED DOMAIN SHIFT DATA
# ==============================================================================

def build_paired_domain_shift(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build paired per-image domain shift dataframe.
    """

    print()

    print(
        "=" * 80
    )

    print(
        "BUILDING PAIRED DOMAIN-SHIFT DATA"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------------------------

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
    )

    source_base = (
        source_base.rename(
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

    # --------------------------------------------------------------------------

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
    )

    target_base = (
        target_base.rename(
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

    # --------------------------------------------------------------------------

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
            "No paired image/model records "
            "were found between Source "
            "and Target domains."
        )

    # --------------------------------------------------------------------------

    merged[
        "degradation"
    ] = (
        calculate_degradation(
            merged[
                "source_f1"
            ].to_numpy(),

            merged[
                "target_f1"
            ].to_numpy(),
        )
    )

    # --------------------------------------------------------------------------

    merged[
        "absolute_f1_drop"
    ] = (

        merged[
            "source_f1"
        ]

        -

        merged[
            "target_f1"
        ]

    )

    # --------------------------------------------------------------------------

    merged[
        "degradation_percent"
    ] = (

        merged[
            "degradation"
        ]

        * 100.0

    )

    # --------------------------------------------------------------------------

    columns = [

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

    ]

    return (
        merged[
            columns
        ]
        .copy()
    )


# ==============================================================================
# DETECT PAIRING MODE
# ==============================================================================

def determine_pairing_mode(
    requested_mode: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Tuple[str, int]:
    """
    Determine paired or unpaired analysis.
    """

    source_ids = set(
        source_df[
            "image_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    target_ids = set(
        target_df[
            "image_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    common_ids = (
        source_ids
        & target_ids
    )

    n_common = (
        len(common_ids)
    )

    if (
        requested_mode
        == "paired"
    ):

        if n_common == 0:

            raise ValueError(
                "Pairing mode is 'paired', "
                "but Source and Target have "
                "no common image_id values."
            )

        return (
            "paired",
            n_common,
        )

    if (
        requested_mode
        == "unpaired"
    ):

        return (
            "unpaired",
            n_common,
        )

    # --------------------------------------------------------------------------
    # AUTO
    # --------------------------------------------------------------------------

    if n_common > 0:

        return (
            "paired",
            n_common,
        )

    return (
        "unpaired",
        0,
    )


# ==============================================================================
# CREATE SUMMARY
# ==============================================================================

def create_domain_shift_summary(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    paired_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Create descriptive summary.
    """

    rows = []

    for model in MODELS:

        source_model = (
            source_df[
                source_df[
                    "model"
                ]
                == model
            ]
            .copy()
        )

        target_model = (
            target_df[
                target_df[
                    "model"
                ]
                == model
            ]
            .copy()
        )

        source_valid = (
            source_model[
                "f1"
            ]
            .dropna()
        )

        target_valid = (
            target_model[
                "f1"
            ]
            .dropna()
        )

        row = {

            "model":
                model,

            "n_source_images":
                len(
                    source_valid
                ),

            "n_target_images":
                len(
                    target_valid
                ),

            "mean_source_f1":
                source_valid.mean(),

            "median_source_f1":
                source_valid.median(),

            "std_source_f1":
                source_valid.std(
                    ddof=1
                ),

            "mean_target_f1":
                target_valid.mean(),

            "median_target_f1":
                target_valid.median(),

            "std_target_f1":
                target_valid.std(
                    ddof=1
                ),

        }

        # --------------------------------------------------------------

        if (
            paired_df
            is not None
        ):

            model_paired = (
                paired_df[
                    paired_df[
                        "model"
                    ]
                    == model
                ]
                .copy()
            )

            degradation = (
                model_paired[
                    "degradation"
                ]
                .dropna()
            )

            row.update(
                {
                    "n_paired_images":
                        len(
                            degradation
                        ),

                    "mean_degradation":
                        degradation.mean(),

                    "median_degradation":
                        degradation.median(),

                    "std_degradation":
                        degradation.std(
                            ddof=1
                        ),

                    "min_degradation":
                        degradation.min(),

                    "max_degradation":
                        degradation.max(),

                    "degraded_images":
                        int(
                            (
                                degradation
                                > 0
                            ).sum()
                        ),

                    "improved_or_equal_images":
                        int(
                            (
                                degradation
                                <= 0
                            ).sum()
                        ),

                }
            )

        else:

            row.update(
                {
                    "n_paired_images":
                        np.nan,

                    "mean_degradation":
                        np.nan,

                    "median_degradation":
                        np.nan,

                    "std_degradation":
                        np.nan,

                    "min_degradation":
                        np.nan,

                    "max_degradation":
                        np.nan,

                    "degraded_images":
                        np.nan,

                    "improved_or_equal_images":
                        np.nan,
                }
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
    )


# ==============================================================================
# INTEGRITY CHECK
# ==============================================================================

def integrity_check(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    paired_df: pd.DataFrame | None,
    n_source_images: int,
    n_target_images: int,
    pairing_mode: str,
) -> Dict[str, Any]:
    """
    Validate H3 output completeness.
    """

    expected_source = (
        n_source_images
        * len(MODELS)
    )

    expected_target = (
        n_target_images
        * len(MODELS)
    )

    actual_source = (
        len(source_df)
    )

    actual_target = (
        len(target_df)
    )

    result = {

        "pairing_mode":
            pairing_mode,

        "n_models":
            len(MODELS),

        "n_source_images":
            n_source_images,

        "n_target_images":
            n_target_images,

        "expected_source_rows":
            expected_source,

        "actual_source_rows":
            actual_source,

        "source_complete":
            (
                expected_source
                == actual_source
            ),

        "expected_target_rows":
            expected_target,

        "actual_target_rows":
            actual_target,

        "target_complete":
            (
                expected_target
                == actual_target
            ),

    }

    if (
        pairing_mode
        == "paired"
        and
        paired_df
        is not None
    ):

        paired_image_ids = (
            paired_df[
                "image_id"
            ]
            .nunique()
        )

        result[
            "n_paired_images"
        ] = (
            paired_image_ids
        )

        result[
            "expected_paired_rows"
        ] = (

            paired_image_ids
            * len(MODELS)

        )

        result[
            "actual_paired_rows"
        ] = (
            len(
                paired_df
            )
        )

        result[
            "paired_complete"
        ] = (

            result[
                "expected_paired_rows"
            ]

            ==

            result[
                "actual_paired_rows"
            ]

        )

    else:

        result[
            "n_paired_images"
        ] = 0

        result[
            "expected_paired_rows"
        ] = 0

        result[
            "actual_paired_rows"
        ] = 0

        result[
            "paired_complete"
        ] = True

    result[
        "all_complete"
    ] = (

        result[
            "source_complete"
        ]

        and

        result[
            "target_complete"
        ]

        and

        result[
            "paired_complete"
        ]

    )

    return (
        result
    )


# ==============================================================================
# JSON SAVE
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
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "H3 Domain-Shift Robustness "
            "Evaluation for YOLOv8, RT-DETR "
            "and RTMDet."
        )
    )

    # --------------------------------------------------------------------------
    # Source
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

    # --------------------------------------------------------------------------
    # Target
    # --------------------------------------------------------------------------

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
    # Models
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
    # Inference
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
    # Pairing
    # --------------------------------------------------------------------------

    parser.add_argument(
        "--pairing-mode",
        choices=[
            "auto",
            "paired",
            "unpaired",
        ],
        default="auto",
    )

    args = (
        parser.parse_args()
    )

    # ==========================================================================
    # HEADER
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "H3 COMPLETE PIPELINE"
    )

    print(
        "DOMAIN-SHIFT ROBUSTNESS EVALUATION"
    )

    print(
        "YOLOv8 vs RT-DETR vs RTMDet"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Source images      : "
        f"{args.source_images}"
    )

    print(
        f"Source labels      : "
        f"{args.source_labels}"
    )

    print()

    print(
        f"Target images      : "
        f"{args.target_images}"
    )

    print(
        f"Target labels      : "
        f"{args.target_labels}"
    )

    print()

    print(
        f"YOLOv8             : "
        f"{args.yolov8}"
    )

    print(
        f"RT-DETR            : "
        f"{args.rtdetr}"
    )

    print(
        f"RTMDet             : "
        f"{args.rtmdet}"
    )

    print(
        f"RTMDet config      : "
        f"{args.rtmdet_config}"
    )

    print()

    print(
        f"Image size         : "
        f"{args.imgsz}"
    )

    print(
        f"Confidence         : "
        f"{args.conf}"
    )

    print(
        f"IoU threshold      : "
        f"{args.iou}"
    )

    print(
        f"Device             : "
        f"{args.device}"
    )

    print(
        f"Pairing mode       : "
        f"{args.pairing_mode}"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        RESULTS_DIR
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
                "\nRequired path "
                "does not exist:\n"
                f"{path}"
            )

    # ==========================================================================
    # LOAD IMAGE SETS
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "LOADING IMAGE SETS"
    )

    print(
        "=" * 80
    )

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

    print(
        "=" * 80
    )

    print(
        "LOADING MODELS"
    )

    print(
        "=" * 80
    )

    from ultralytics import YOLO

    # --------------------------------------------------------------------------

    print()

    print(
        "Loading YOLOv8..."
    )

    yolo_model = (
        YOLO(
            str(
                args.yolov8
            )
        )
    )

    # --------------------------------------------------------------------------

    print(
        "Loading RT-DETR..."
    )

    rtdetr_model = (
        YOLO(
            str(
                args.rtdetr
            )
        )
    )

    # --------------------------------------------------------------------------

    print(
        "Loading RTMDet..."
    )

    from mmdet.apis import (
        init_detector
    )

    rtmdet_model = (
        init_detector(
            str(
                args.rtmdet_config
            ),

            str(
                args.rtmdet
            ),

            device=
                args.device,
        )
    )

    # ==========================================================================
    # SOURCE EVALUATION
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "STEP 1/3 - SOURCE DOMAIN"
    )

    print(
        "=" * 80
    )

    source_rows = []

    # --------------------------------------------------------------------------

    source_rows.extend(
        evaluate_model(

            model_name=
                "YOLOv8",

            model=
                yolo_model,

            image_paths=
                source_images,

            labels_dir=
                args.source_labels,

            domain_name=
                "source",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    # --------------------------------------------------------------------------

    source_rows.extend(
        evaluate_model(

            model_name=
                "RT-DETR",

            model=
                rtdetr_model,

            image_paths=
                source_images,

            labels_dir=
                args.source_labels,

            domain_name=
                "source",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    # --------------------------------------------------------------------------

    source_rows.extend(
        evaluate_model(

            model_name=
                "RTMDet",

            model=
                rtmdet_model,

            image_paths=
                source_images,

            labels_dir=
                args.source_labels,

            domain_name=
                "source",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    source_df = (
        pd.DataFrame(
            source_rows
        )
    )

    source_output = (
        RESULTS_DIR
        / "source_per_image_metrics.csv"
    )

    source_df.to_csv(
        source_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print(
        "Source metrics saved:"
    )

    print(
        source_output
    )

    # ==========================================================================
    # TARGET EVALUATION
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "STEP 2/3 - TARGET DOMAIN"
    )

    print(
        "=" * 80
    )

    target_rows = []

    # --------------------------------------------------------------------------

    target_rows.extend(
        evaluate_model(

            model_name=
                "YOLOv8",

            model=
                yolo_model,

            image_paths=
                target_images,

            labels_dir=
                args.target_labels,

            domain_name=
                "target",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    # --------------------------------------------------------------------------

    target_rows.extend(
        evaluate_model(

            model_name=
                "RT-DETR",

            model=
                rtdetr_model,

            image_paths=
                target_images,

            labels_dir=
                args.target_labels,

            domain_name=
                "target",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    # --------------------------------------------------------------------------

    target_rows.extend(
        evaluate_model(

            model_name=
                "RTMDet",

            model=
                rtmdet_model,

            image_paths=
                target_images,

            labels_dir=
                args.target_labels,

            domain_name=
                "target",

            imgsz=
                args.imgsz,

            conf=
                args.conf,

            iou_threshold=
                args.iou,

            device=
                args.device,

        )
    )

    target_df = (
        pd.DataFrame(
            target_rows
        )
    )

    target_output = (
        RESULTS_DIR
        / "target_per_image_metrics.csv"
    )

    target_df.to_csv(
        target_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print(
        "Target metrics saved:"
    )

    print(
        target_output
    )

    # ==========================================================================
    # DUPLICATE CHECK
    # ==========================================================================

    check_duplicate_ids(
        source_df,
        "Source dataframe",
    )

    check_duplicate_ids(
        target_df,
        "Target dataframe",
    )

    # ==========================================================================
    # REMOVE FAILED ROWS FOR PAIRING DETECTION
    # ==========================================================================

    source_valid = (
        source_df.dropna(
            subset=[
                "image_id",
                "model",
                "f1",
            ]
        )
        .copy()
    )

    target_valid = (
        target_df.dropna(
            subset=[
                "image_id",
                "model",
                "f1",
            ]
        )
        .copy()
    )

    # ==========================================================================
    # DETERMINE PAIRING MODE
    # ==========================================================================

    pairing_mode, n_common = (
        determine_pairing_mode(

            requested_mode=
                args.pairing_mode,

            source_df=
                source_valid,

            target_df=
                target_valid,

        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "PAIRING ANALYSIS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Effective pairing mode: "
        f"{pairing_mode}"
    )

    print(
        f"Common image IDs: "
        f"{n_common:,}"
    )

    # ==========================================================================
    # PAIRED DOMAIN SHIFT
    # ==========================================================================

    paired_df = None

    paired_output = (
        RESULTS_DIR
        / "per_image_domain_shift.csv"
    )

    if (
        pairing_mode
        == "paired"
    ):

        print()

        print(
            "=" * 80
        )

        print(
            "STEP 3/3 - DOMAIN-SHIFT "
            "DEGRADATION"
        )

        print(
            "=" * 80
        )

        paired_df = (
            build_paired_domain_shift(

                source_df=
                    source_valid,

                target_df=
                    target_valid,

            )
        )

        paired_df.to_csv(
            paired_output,
            index=False,
            encoding="utf-8-sig",
        )

        print()

        print(
            "Per-image domain shift saved:"
        )

        print(
            paired_output
        )

    else:

        print()

        print(
            "UNPAIRED MODE DETECTED."
        )

        print(
            "Per-image degradation "
            "was not calculated."
        )

        print(
            "Source and Target images "
            "are treated as independent."
        )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    summary_df = (
        create_domain_shift_summary(

            source_df=
                source_valid,

            target_df=
                target_valid,

            paired_df=
                paired_df,

        )
    )

    summary_output = (
        RESULTS_DIR
        / "domain_shift_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    # ==========================================================================
    # INTEGRITY CHECK
    # ==========================================================================

    integrity = (
        integrity_check(

            source_df=
                source_df,

            target_df=
                target_df,

            paired_df=
                paired_df,

            n_source_images=
                len(
                    source_images
                ),

            n_target_images=
                len(
                    target_images
                ),

            pairing_mode=
                pairing_mode,

        )
    )

    integrity_output = (
        RESULTS_DIR
        / "H3_integrity_check.json"
    )

    save_json(
        integrity,
        integrity_output,
    )

    # ==========================================================================
    # RUN CONFIGURATION
    # ==========================================================================

    run_config = {

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

        "yolov8":
            str(
                args.yolov8
            ),

        "rtdetr":
            str(
                args.rtdetr
            ),

        "rtmdet":
            str(
                args.rtmdet
            ),

        "rtmdet_config":
            str(
                args.rtmdet_config
            ),

        "imgsz":
            args.imgsz,

        "confidence":
            args.conf,

        "iou":
            args.iou,

        "device":
            args.device,

        "requested_pairing_mode":
            args.pairing_mode,

        "effective_pairing_mode":
            pairing_mode,

        "common_image_ids":
            n_common,

        "models":
            MODELS,

        "n_source_images":
            len(
                source_images
            ),

        "n_target_images":
            len(
                target_images
            ),

    }

    config_output = (
        RESULTS_DIR
        / "H3_run_configuration.json"
    )

    save_json(
        run_config,
        config_output,
    )

    # ==========================================================================
    # CONSOLE SUMMARY
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "DOMAIN-SHIFT SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ==========================================================================
    # INTEGRITY REPORT
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "INTEGRITY CHECK"
    )

    print(
        "=" * 80
    )

    print()

    for key, value in (
        integrity.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ==========================================================================
    # FINAL
    # ==========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "H3 COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Generated files:"
    )

    print()

    print(
        f"1. {source_output}"
    )

    print(
        f"2. {target_output}"
    )

    if (
        paired_df
        is not None
    ):

        print(
            f"3. {paired_output}"
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

    if integrity[
        "all_complete"
    ]:

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

        sys.exit(
            130
        )

    except Exception as exc:

        print()

        print(
            "=" * 80
        )

        print(
            "FATAL ERROR"
        )

        print(
            "=" * 80
        )

        print(
            str(exc)
        )

        import traceback

        traceback.print_exc()

        sys.exit(
            1
        )

