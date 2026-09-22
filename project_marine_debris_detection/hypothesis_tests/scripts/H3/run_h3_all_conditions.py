# -*- coding: utf-8 -*-

"""
================================================================================
H3 ALL-CONDITIONS AGGREGATOR
================================================================================

Purpose:
    Run all 12 simulated domain-shift conditions while preserving every
    condition in one combined dataset.

Important:
    - Source inference is performed ONCE.
    - Models are loaded ONCE.
    - Target inference is performed once per condition.
    - Results from different conditions are NEVER overwritten.
    - Final paired dataset contains:

        condition
        image_id
        model
        source_f1
        target_f1
        absolute_f1_drop
        degradation
        degradation_percent

Expected rows:

    12 conditions × 2462 images × 3 models
    = 88,632 rows

================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ==============================================================================
# PATHS
# ==============================================================================

SCRIPT_PATH = Path(__file__).resolve()

HYPOTHESIS_TESTS_DIR = SCRIPT_PATH.parents[2]
PROJECT = HYPOTHESIS_TESTS_DIR.parent

RESULTS_DIR = (
    HYPOTHESIS_TESTS_DIR
    / "results"
    / "H3"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMBINED_OUTPUT = (
    RESULTS_DIR
    / "H3_all_conditions_per_image_domain_shift.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "H3_all_conditions_summary.csv"
)

INTEGRITY_OUTPUT = (
    RESULTS_DIR
    / "H3_all_conditions_integrity.csv"
)


# ==============================================================================
# EXISTING H3 PIPELINE
# ==============================================================================

CURRENT_H3_SCRIPT = (
    Path(__file__).parent
    / "run_h3_complete.py"
)

sys.path.insert(
    0,
    str(CURRENT_H3_SCRIPT.parent),
)

import run_h3_complete as h3


# ==============================================================================
# CONDITIONS
# ==============================================================================

CONDITIONS = [
    "low_light_extreme",
    "high_light_extreme",
    "gamma_dark",
    "gamma_bright",
    "color_temperature_cool",
    "color_temperature_warm",
    "rotation_minus_15",
    "rotation_plus_15",
    "perspective_direction",
    "shear_direction",
    "scale_view_shift",
    "uav_coastal_domain_shift",
]


MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

SOURCE_IMAGES = Path(
    r"E:\rename\yolo_work_out\resrcher\YOLO_DATASETS\trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED\images\test"
)

SOURCE_LABELS = Path(
    r"E:\rename\yolo_work_out\resrcher\YOLO_DATASETS\trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED\labels\test"
)

DOMAIN_ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection\hypothesis_tests\data\domain_shift"
)


# ==============================================================================
# MODEL PATHS
# ==============================================================================

YOLOV8_PATH = Path(
    r"E:\rename\yolo_work_out\resrcher\yolo_train\results\yolov8n\checkpoints\yolov8\weights\best.pt"
)

RTDETR_PATH = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection\runs\rtdetr\rtdetr_l_640_6gb_diskcache\weights\last.pt"
)

RTMDET_PATH = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection\runs\rtmdet\rtmdet_tiny_640_6gb\best_coco_bbox_mAP_epoch_146.pth"
)

RTMDET_CONFIG = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection\runs\rtmdet\rtmdet_tiny_640_6gb\rtmdet_tiny_marine_debris_config.py"
)


# ==============================================================================
# SETTINGS
# ==============================================================================

IMGSZ = 640
CONF = 0.25
IOU = 0.50
DEVICE = "cuda:0"


# ==============================================================================
# VALIDATE PATHS
# ==============================================================================

def validate_paths():

    paths = [
        SOURCE_IMAGES,
        SOURCE_LABELS,
        DOMAIN_ROOT,
        YOLOV8_PATH,
        RTDETR_PATH,
        RTMDET_PATH,
        RTMDET_CONFIG,
    ]

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path does not exist:\n{path}"
            )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print()
    print("=" * 80)
    print("H3 ALL-CONDITIONS PIPELINE")
    print("=" * 80)

    print()
    print(f"Conditions : {len(CONDITIONS)}")
    print(f"Models     : {len(MODELS)}")
    print(f"Image size : {IMGSZ}")
    print(f"Confidence : {CONF}")
    print(f"IoU        : {IOU}")
    print(f"Device     : {DEVICE}")

    validate_paths()

    # ==========================================================================
    # LOAD IMAGE SETS
    # ==========================================================================

    print()
    print("=" * 80)
    print("LOADING SOURCE DATA")
    print("=" * 80)

    source_images = h3.validate_image_set(
        SOURCE_IMAGES
    )

    print()
    print(
        f"Source images: {len(source_images):,}"
    )

    if len(source_images) != 2462:

        raise RuntimeError(
            "Expected 2462 source images, "
            f"but found {len(source_images)}."
        )

    # ==========================================================================
    # LOAD MODELS ONCE
    # ==========================================================================

    print()
    print("=" * 80)
    print("LOADING MODELS ONCE")
    print("=" * 80)

    from ultralytics import YOLO

    print()
    print("Loading YOLOv8...")
    yolo_model = YOLO(
        str(YOLOV8_PATH)
    )

    print("Loading RT-DETR...")
    rtdetr_model = YOLO(
        str(RTDETR_PATH)
    )

    print("Loading RTMDet...")

    from mmdet.apis import init_detector

    rtmdet_model = init_detector(
        str(RTMDET_CONFIG),
        str(RTMDET_PATH),
        device=DEVICE,
    )

    # ==========================================================================
    # SOURCE EVALUATION — ONCE ONLY
    # ==========================================================================

    print()
    print("=" * 80)
    print("SOURCE DOMAIN EVALUATION")
    print("IMPORTANT: SOURCE IS EVALUATED ONLY ONCE")
    print("=" * 80)

    source_rows = []

    print()
    print("YOLOv8 - Source")

    source_rows.extend(
        h3.evaluate_model(
            model_name="YOLOv8",
            model=yolo_model,
            image_paths=source_images,
            labels_dir=SOURCE_LABELS,
            domain_name="source",
            imgsz=IMGSZ,
            conf=CONF,
            iou_threshold=IOU,
            device=DEVICE,
        )
    )

    print()
    print("RT-DETR - Source")

    source_rows.extend(
        h3.evaluate_model(
            model_name="RT-DETR",
            model=rtdetr_model,
            image_paths=source_images,
            labels_dir=SOURCE_LABELS,
            domain_name="source",
            imgsz=IMGSZ,
            conf=CONF,
            iou_threshold=IOU,
            device=DEVICE,
        )
    )

    print()
    print("RTMDet - Source")

    source_rows.extend(
        h3.evaluate_model(
            model_name="RTMDet",
            model=rtmdet_model,
            image_paths=source_images,
            labels_dir=SOURCE_LABELS,
            domain_name="source",
            imgsz=IMGSZ,
            conf=CONF,
            iou_threshold=IOU,
            device=DEVICE,
        )
    )

    source_df = pd.DataFrame(
        source_rows
    )

    source_df = source_df.dropna(
        subset=[
            "image_id",
            "model",
            "f1",
        ]
    ).copy()

    print()
    print(
        f"Source rows: {len(source_df):,}"
    )

    expected_source_rows = (
        2462 * 3
    )

    if len(source_df) != expected_source_rows:

        raise RuntimeError(
            "Source evaluation incomplete.\n"
            f"Expected: {expected_source_rows:,}\n"
            f"Actual:   {len(source_df):,}"
        )

    # ==========================================================================
    # PREPARE SOURCE FOR MERGE
    # ==========================================================================

    source_base = source_df[
        [
            "image_id",
            "model",
            "precision",
            "recall",
            "f1",
        ]
    ].copy()

    source_base.rename(
        columns={
            "precision": "source_precision",
            "recall": "source_recall",
            "f1": "source_f1",
        },
        inplace=True,
    )

    # ==========================================================================
    # TARGET CONDITIONS
    # ==========================================================================

    all_condition_results = []

    summary_rows = []

    for condition_index, condition in enumerate(
        CONDITIONS,
        start=1,
    ):

        print()
        print()
        print("#" * 80)
        print(
            f"CONDITION {condition_index}/{len(CONDITIONS)}"
        )
        print(
            f"TARGET: {condition}"
        )
        print("#" * 80)

        target_images_dir = (
            DOMAIN_ROOT
            / condition
            / "images"
        )

        target_labels_dir = (
            DOMAIN_ROOT
            / condition
            / "labels"
        )

        if not target_images_dir.exists():

            raise FileNotFoundError(
                f"Target image directory does not exist:\n"
                f"{target_images_dir}"
            )

        if not target_labels_dir.exists():

            raise FileNotFoundError(
                f"Target label directory does not exist:\n"
                f"{target_labels_dir}"
            )

        target_images = h3.validate_image_set(
            target_images_dir
        )

        print()
        print(
            f"Target images: {len(target_images):,}"
        )

        if len(target_images) != 2462:

            raise RuntimeError(
                f"Condition '{condition}' "
                f"has {len(target_images)} images; "
                f"expected 2462."
            )

        # ----------------------------------------------------------------------
        # TARGET — YOLO
        # ----------------------------------------------------------------------

        target_rows = []

        print()
        print(
            f"[{condition}] YOLOv8"
        )

        target_rows.extend(
            h3.evaluate_model(
                model_name="YOLOv8",
                model=yolo_model,
                image_paths=target_images,
                labels_dir=target_labels_dir,
                domain_name=condition,
                imgsz=IMGSZ,
                conf=CONF,
                iou_threshold=IOU,
                device=DEVICE,
            )
        )

        # ----------------------------------------------------------------------
        # TARGET — RT-DETR
        # ----------------------------------------------------------------------

        print()
        print(
            f"[{condition}] RT-DETR"
        )

        target_rows.extend(
            h3.evaluate_model(
                model_name="RT-DETR",
                model=rtdetr_model,
                image_paths=target_images,
                labels_dir=target_labels_dir,
                domain_name=condition,
                imgsz=IMGSZ,
                conf=CONF,
                iou_threshold=IOU,
                device=DEVICE,
            )
        )

        # ----------------------------------------------------------------------
        # TARGET — RTMDet
        # ----------------------------------------------------------------------

        print()
        print(
            f"[{condition}] RTMDet"
        )

        target_rows.extend(
            h3.evaluate_model(
                model_name="RTMDet",
                model=rtmdet_model,
                image_paths=target_images,
                labels_dir=target_labels_dir,
                domain_name=condition,
                imgsz=IMGSZ,
                conf=CONF,
                iou_threshold=IOU,
                device=DEVICE,
            )
        )

        target_df = pd.DataFrame(
            target_rows
        )

        target_df = target_df.dropna(
            subset=[
                "image_id",
                "model",
                "f1",
            ]
        ).copy()

        expected_target_rows = (
            2462 * 3
        )

        if len(target_df) != expected_target_rows:

            raise RuntimeError(
                f"Condition '{condition}' incomplete.\n"
                f"Expected: {expected_target_rows:,}\n"
                f"Actual:   {len(target_df):,}"
            )

        # ----------------------------------------------------------------------
        # TARGET BASE
        # ----------------------------------------------------------------------

        target_base = target_df[
            [
                "image_id",
                "model",
                "precision",
                "recall",
                "f1",
            ]
        ].copy()

        target_base.rename(
            columns={
                "precision": "target_precision",
                "recall": "target_recall",
                "f1": "target_f1",
            },
            inplace=True,
        )

        # ----------------------------------------------------------------------
        # MERGE SOURCE + TARGET
        # ----------------------------------------------------------------------

        merged = source_base.merge(
            target_base,
            on=[
                "image_id",
                "model",
            ],
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != expected_target_rows:

            raise RuntimeError(
                f"Condition '{condition}' pairing error.\n"
                f"Expected paired rows: {expected_target_rows:,}\n"
                f"Actual paired rows:   {len(merged):,}"
            )

        # ----------------------------------------------------------------------
        # DEGRADATION
        # ----------------------------------------------------------------------

        merged[
            "absolute_f1_drop"
        ] = (
            merged["source_f1"]
            -
            merged["target_f1"]
        )

        merged[
            "degradation"
        ] = h3.calculate_degradation(
            merged["source_f1"].to_numpy(),
            merged["target_f1"].to_numpy(),
        )

        merged[
            "degradation_percent"
        ] = (
            merged["degradation"]
            * 100.0
        )

        # ----------------------------------------------------------------------
        # ADD CONDITION
        # ----------------------------------------------------------------------

        merged.insert(
            0,
            "condition",
            condition,
        )

        # ----------------------------------------------------------------------
        # ORDER COLUMNS
        # ----------------------------------------------------------------------

        merged = merged[
            [
                "condition",
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
        ].copy()

        all_condition_results.append(
            merged
        )

        # ----------------------------------------------------------------------
        # CONDITION SUMMARY
        # ----------------------------------------------------------------------

        for model in MODELS:

            model_df = merged[
                merged["model"] == model
            ]

            degradation = (
                model_df["degradation"]
                .dropna()
            )

            summary_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "n_images": len(model_df),
                    "mean_source_f1":
                        model_df[
                            "source_f1"
                        ].mean(),
                    "mean_target_f1":
                        model_df[
                            "target_f1"
                        ].mean(),
                    "mean_degradation":
                        degradation.mean(),
                    "median_degradation":
                        degradation.median(),
                    "std_degradation":
                        degradation.std(
                            ddof=1
                        ),
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

        print()
        print(
            "=" * 80
        )
        print(
            f"CONDITION COMPLETED: {condition}"
        )
        print(
            "=" * 80
        )

        print()
        print(
            merged.groupby(
                "model"
            )[
                "degradation"
            ]
            .agg(
                [
                    "mean",
                    "median",
                    "std",
                ]
            )
            .to_string()
        )

    # ==========================================================================
    # COMBINE ALL CONDITIONS
    # ==========================================================================

    print()
    print()
    print("=" * 80)
    print("COMBINING ALL H3 CONDITIONS")
    print("=" * 80)

    combined_df = pd.concat(
        all_condition_results,
        ignore_index=True,
    )

    expected_total_rows = (
        len(CONDITIONS)
        * 2462
        * 3
    )

    actual_total_rows = len(
        combined_df
    )

    print()
    print(
        f"Expected rows: {expected_total_rows:,}"
    )

    print(
        f"Actual rows  : {actual_total_rows:,}"
    )

    if actual_total_rows != expected_total_rows:

        raise RuntimeError(
            "FINAL H3 DATASET SIZE ERROR.\n"
            f"Expected: {expected_total_rows:,}\n"
            f"Actual:   {actual_total_rows:,}"
        )

    # ==========================================================================
    # DUPLICATE CHECK
    # ==========================================================================

    duplicates = combined_df[
        combined_df.duplicated(
            subset=[
                "condition",
                "image_id",
                "model",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:

        print()
        print(
            "DUPLICATES DETECTED:"
        )

        print(
            duplicates.head(20).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate condition/image/model "
            "records detected."
        )

    # ==========================================================================
    # CONDITION COUNT CHECK
    # ==========================================================================

    condition_counts = (
        combined_df
        .groupby(
            "condition"
        )
        .size()
        .sort_index()
    )

    print()
    print(
        "Rows per condition:"
    )

    print(
        condition_counts.to_string()
    )

    expected_per_condition = (
        2462 * 3
    )

    if not (
        condition_counts
        == expected_per_condition
    ).all():

        raise RuntimeError(
            "At least one H3 condition "
            "does not contain the expected "
            "7386 rows."
        )

    # ==========================================================================
    # MODEL COUNT CHECK
    # ==========================================================================

    model_counts = (
        combined_df
        .groupby(
            "model"
        )
        .size()
        .sort_index()
    )

    print()
    print(
        "Rows per model:"
    )

    print(
        model_counts.to_string()
    )

    expected_per_model = (
        len(CONDITIONS)
        * 2462
    )

    if not (
        model_counts
        == expected_per_model
    ).all():

        raise RuntimeError(
            "Model row count is incorrect."
        )

    # ==========================================================================
    # SAVE FINAL COMBINED DATASET
    # ==========================================================================

    combined_df.to_csv(
        COMBINED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "FINAL H3 DATASET SAVED:"
    )

    print(
        COMBINED_OUTPUT
    )

    # ==========================================================================
    # SAVE SUMMARY
    # ==========================================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "H3 CONDITION SUMMARY SAVED:"
    )

    print(
        SUMMARY_OUTPUT
    )

    # ==========================================================================
    # INTEGRITY TABLE
    # ==========================================================================

    integrity_rows = []

    for condition in CONDITIONS:

        condition_df = combined_df[
            combined_df["condition"]
            == condition
        ]

        integrity_rows.append(
            {
                "condition": condition,
                "expected_rows":
                    2462 * 3,
                "actual_rows":
                    len(condition_df),
                "complete":
                    len(condition_df)
                    == 2462 * 3,
                "unique_images":
                    condition_df[
                        "image_id"
                    ].nunique(),
                "unique_models":
                    condition_df[
                        "model"
                    ].nunique(),
            }
        )

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    integrity_df.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "H3 INTEGRITY REPORT SAVED:"
    )

    print(
        INTEGRITY_OUTPUT
    )

    # ==========================================================================
    # FINAL REPORT
    # ==========================================================================

    print()
    print("=" * 80)
    print("H3 ALL-CONDITIONS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Conditions       : {len(CONDITIONS)}"
    )

    print(
        f"Images/condition : 2,462"
    )

    print(
        f"Models           : 3"
    )

    print(
        f"Expected rows    : {expected_total_rows:,}"
    )

    print(
        f"Actual rows      : {actual_total_rows:,}"
    )

    print()

    if actual_total_rows == expected_total_rows:

        print(
            "FINAL STATUS: H3 DATASET COMPLETE"
        )

    else:

        print(
            "FINAL STATUS: H3 DATASET INCOMPLETE"
        )

    print()
    print(
        "Output:"
    )

    print(
        COMBINED_OUTPUT
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
        print()
        print(str(exc))

        import traceback

        traceback.print_exc()

        sys.exit(1)