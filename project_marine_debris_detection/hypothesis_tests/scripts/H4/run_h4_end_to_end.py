# -*- coding: utf-8 -*-

"""
================================================================================
H4 - Synthetic Domain Adaptation for YOLOv8
================================================================================

هدف:
    1) انتخاب YOLOv8 به‌عنوان مدل منتخب H4 بر اساس H1/H2/H3
    2) ساخت یک Synthetic Target Domain فقط از Train/Validation
    3) آموزش YOLOv8 روی داده Synthetic Target Domain
    4) ارزیابی مدل قبل و بعد از Adaptation روی Final Test Set
    5) محاسبه آزمون‌های آماری H4
    6) تولید نمودارها و فایل‌های گزارش

نکات روش‌شناختی:
    - Final Test Set به هیچ عنوان برای Adaptation استفاده نمی‌شود.
    - img size = 640 ثابت است.
    - confidence = 0.25
    - IoU matching = 0.50
    - seed = 42
    - Adaptation کاملاً Synthetic / Controlled است.
    - این آزمایش نباید به‌عنوان Real External Target-Domain Generalization
      تفسیر شود.
================================================================================
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    wilcoxon,
    spearmanr,
)

from sklearn.utils import resample

from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
)

DATASET_ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\YOLO_DATASETS"
    r"\trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED"
)

BASE_MODEL = Path(
    r"E:\rename\yolo_work_out\resrcher\yolo_train\results"
    r"\yolov8n\checkpoints\yolov8\weights\best.pt"
)

H4_ROOT = PROJECT_ROOT / "hypothesis_tests"

ADAPT_DATASET_ROOT = (
    H4_ROOT
    / "data"
    / "H4"
    / "adaptation_dataset"
)

H4_RUN_ROOT = (
    H4_ROOT
    / "runs"
    / "H4"
)

H4_RESULTS_ROOT = (
    H4_ROOT
    / "results"
    / "H4"
)

H4_STATS_ROOT = (
    H4_ROOT
    / "results"
    / "statistics"
    / "H4"
)

H4_FIGURES_ROOT = (
    H4_ROOT
    / "results"
    / "figures"
    / "H4"
)


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

IMAGE_SIZE = 640

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

SEED = 42
DEVICE = 0

ADAPT_EPOCHS = 20

# افزایش سرعت نسبت به نسخه قبلی
ADAPT_BATCH = 10

# تعداد worker بیشتر برای DataLoader
ADAPT_WORKERS = 4

# AMP برای RTX 3060
ADAPT_AMP = False

# استفاده از disk cache
ADAPT_CACHE = "disk"

ADAPT_LR = 1e-4

ADAPT_PATIENCE = 10

MAX_TRAIN_IMAGES = 3000
MAX_VAL_IMAGES = 500

JPEG_QUALITY = 90


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass


# =============================================================================
# DIRECTORY HELPERS
# =============================================================================

def create_directories() -> None:

    for p in [
        ADAPT_DATASET_ROOT,
        H4_RUN_ROOT,
        H4_RESULTS_ROOT,
        H4_STATS_ROOT,
        H4_FIGURES_ROOT,
    ]:
        p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMAGE HELPERS
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(folder: Path) -> List[Path]:

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder does not exist:\n{folder}"
        )

    files = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(files)


def corresponding_label(
    image_path: Path,
    label_dir: Path,
) -> Path:

    return label_dir / f"{image_path.stem}.txt"


# =============================================================================
# SYNTHETIC DOMAIN TRANSFORMATION
# =============================================================================

def apply_synthetic_target_transform(
    image: np.ndarray,
) -> np.ndarray:

    """
    Controlled synthetic target-domain transform.

    بدون تغییر هندسی:
        - low illumination
        - gamma
        - contrast
        - color temperature
        - mild blur
        - JPEG compression

    چون transform هندسی نداریم، Bounding Box ها بدون تغییر باقی می‌مانند.
    """

    img = image.astype(np.float32) / 255.0

    # -------------------------------------------------------------------------
    # 1. Low illumination
    # -------------------------------------------------------------------------

    img *= 0.62

    # -------------------------------------------------------------------------
    # 2. Gamma
    # -------------------------------------------------------------------------

    gamma = 1.18

    img = np.clip(img, 0.0, 1.0)

    img = np.power(
        img,
        gamma
    )

    # -------------------------------------------------------------------------
    # 3. Contrast reduction
    # -------------------------------------------------------------------------

    contrast = 0.82

    img = (
        img - 0.5
    ) * contrast + 0.5

    # -------------------------------------------------------------------------
    # 4. Warm/cool color shift
    # -------------------------------------------------------------------------

    # OpenCV = BGR
    img[:, :, 0] *= 1.08  # Blue
    img[:, :, 1] *= 1.00  # Green
    img[:, :, 2] *= 0.92  # Red

    img = np.clip(
        img,
        0.0,
        1.0
    )

    img = (
        img * 255.0
    ).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 5. Mild blur
    # -------------------------------------------------------------------------

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    # -------------------------------------------------------------------------
    # 6. JPEG compression
    # -------------------------------------------------------------------------

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY,
        ],
    )

    if success:
        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if decoded is not None:
            img = decoded

    return img


# =============================================================================
# COPY LABEL
# =============================================================================

def copy_label(
    source_label: Path,
    target_label: Path,
) -> None:

    if source_label.exists():
        shutil.copy2(
            source_label,
            target_label,
        )
    else:
        target_label.write_text(
            "",
            encoding="utf-8",
        )


# =============================================================================
# BUILD ADAPTATION DATASET
# =============================================================================

def build_adaptation_dataset() -> Dict:

    print("\n" + "=" * 80)
    print("BUILDING SYNTHETIC H4 ADAPTATION DATASET")
    print("=" * 80)

    source_train_images = (
        DATASET_ROOT
        / "images"
        / "train"
    )

    source_train_labels = (
        DATASET_ROOT
        / "labels"
        / "train"
    )

    source_val_images = (
        DATASET_ROOT
        / "images"
        / "val"
    )

    source_val_labels = (
        DATASET_ROOT
        / "labels"
        / "val"
    )

    train_images = list_images(
        source_train_images
    )

    val_images = list_images(
        source_val_images
    )

    rng = random.Random(SEED)

    rng.shuffle(train_images)
    rng.shuffle(val_images)

    train_images = train_images[
        : min(
            MAX_TRAIN_IMAGES,
            len(train_images),
        )
    ]

    val_images = val_images[
        : min(
            MAX_VAL_IMAGES,
            len(val_images),
        )
    ]

    print(
        f"Selected adaptation train images: "
        f"{len(train_images)}"
    )

    print(
        f"Selected adaptation val images: "
        f"{len(val_images)}"
    )

    # -------------------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------------------

    train_img_out = (
        ADAPT_DATASET_ROOT
        / "images"
        / "train"
    )

    train_lbl_out = (
        ADAPT_DATASET_ROOT
        / "labels"
        / "train"
    )

    val_img_out = (
        ADAPT_DATASET_ROOT
        / "images"
        / "val"
    )

    val_lbl_out = (
        ADAPT_DATASET_ROOT
        / "labels"
        / "val"
    )

    for p in [
        train_img_out,
        train_lbl_out,
        val_img_out,
        val_lbl_out,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------------------
    # Process function
    # -------------------------------------------------------------------------

    def process_split(
        images: List[Path],
        source_label_dir: Path,
        target_image_dir: Path,
        target_label_dir: Path,
        split_name: str,
    ) -> int:

        processed = 0

        for idx, image_path in enumerate(images):

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                warnings.warn(
                    f"Could not read image: {image_path}"
                )
                continue

            transformed = apply_synthetic_target_transform(
                image
            )

            target_image = (
                target_image_dir
                / image_path.name
            )

            target_label = (
                target_label_dir
                / f"{image_path.stem}.txt"
            )

            cv2.imwrite(
                str(target_image),
                transformed,
            )

            source_label = corresponding_label(
                image_path,
                source_label_dir,
            )

            copy_label(
                source_label,
                target_label,
            )

            processed += 1

            if (
                (idx + 1) % 250 == 0
                or idx == len(images) - 1
            ):
                print(
                    f"[{split_name}] "
                    f"{idx + 1}/{len(images)}"
                )

        return processed

    train_count = process_split(
        train_images,
        source_train_labels,
        train_img_out,
        train_lbl_out,
        "train",
    )

    val_count = process_split(
        val_images,
        source_val_labels,
        val_img_out,
        val_lbl_out,
        "val",
    )

    # -------------------------------------------------------------------------
    # Create YOLO YAML
    # -------------------------------------------------------------------------

    source_yaml = (
        DATASET_ROOT
        / "data.yaml"
    )

    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Original data.yaml not found:\n{source_yaml}"
        )

    # خواندن کلاس‌ها از YAML
    import yaml

    with open(
        source_yaml,
        "r",
        encoding="utf-8",
    ) as f:
        yaml_data = yaml.safe_load(f)

    names = yaml_data.get(
        "names",
        []
    )

    adaptation_yaml = (
        ADAPT_DATASET_ROOT
        / "data.yaml"
    )

    yaml_content = {
        "path": str(
            ADAPT_DATASET_ROOT
        ),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": len(names),
    }

    with open(
        adaptation_yaml,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            yaml_content,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    summary = {
        "seed": SEED,
        "source_dataset": str(DATASET_ROOT),
        "adaptation_dataset": str(
            ADAPT_DATASET_ROOT
        ),
        "train_images": train_count,
        "val_images": val_count,
        "transform": {
            "low_illumination": True,
            "gamma": 1.18,
            "contrast": 0.82,
            "color_temperature_shift": True,
            "mild_gaussian_blur": True,
            "jpeg_compression": JPEG_QUALITY,
            "geometric_transform": False,
        },
        "final_test_used_for_adaptation": False,
    }

    with open(
        H4_RESULTS_ROOT
        / "H4_adaptation_dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nAdaptation dataset created successfully.")

    return summary


# =============================================================================
# TRAIN YOLOV8 ADAPTATION MODEL
# =============================================================================

def train_adapted_model() -> Path:

    print("\n" + "=" * 80)
    print("TRAINING YOLOv8 SYNTHETIC ADAPTATION MODEL")
    print("=" * 80)

    adaptation_yaml = (
        ADAPT_DATASET_ROOT
        / "data.yaml"
    )

    if not adaptation_yaml.exists():
        raise FileNotFoundError(
            f"Adaptation data.yaml missing:\n{adaptation_yaml}"
        )

    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            f"Base YOLOv8 checkpoint missing:\n{BASE_MODEL}"
        )

    print(f"Base model: {BASE_MODEL}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {ADAPT_BATCH}")
    print(f"Workers: {ADAPT_WORKERS}")
    print(f"AMP: {ADAPT_AMP}")
    print(f"Cache: {ADAPT_CACHE}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {ADAPT_EPOCHS}")

    model = YOLO(
        str(BASE_MODEL)
    )

    results = model.train(

        # ---------------------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------------------

        data=str(
            adaptation_yaml
        ),

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------

        epochs=ADAPT_EPOCHS,

        batch=ADAPT_BATCH,

        imgsz=IMAGE_SIZE,

        # ---------------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------------

        optimizer="AdamW",

        lr0=ADAPT_LR,

        # ---------------------------------------------------------------------
        # Reproducibility
        # ---------------------------------------------------------------------

        seed=SEED,

        deterministic=True,

        # ---------------------------------------------------------------------
        # GPU
        # ---------------------------------------------------------------------

        device=DEVICE,

        # ---------------------------------------------------------------------
        # SPEED SETTINGS
        # ---------------------------------------------------------------------

        workers=ADAPT_WORKERS,

        cache=ADAPT_CACHE,

        amp=ADAPT_AMP,

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        patience=ADAPT_PATIENCE,

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        project=str(
            H4_RUN_ROOT
        ),

        name="yolov8_synthetic_adaptation",

        exist_ok=True,

        save=True,

        save_period=1,

        plots=True,

        verbose=True,

    )

    save_dir = Path(
        results.save_dir
    )

    best_checkpoint = (
        save_dir
        / "weights"
        / "best.pt"
    )

    if not best_checkpoint.exists():

        raise FileNotFoundError(
            "Adapted best.pt was not created:\n"
            f"{best_checkpoint}"
        )

    print(
        "\nAdapted model created:"
    )

    print(
        best_checkpoint
    )

    return best_checkpoint


# =============================================================================
# LOAD YOLO MODEL
# =============================================================================

def load_model(
    checkpoint: Path,
) -> YOLO:

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    return YOLO(
        str(checkpoint)
    )


# =============================================================================
# READ YOLO LABELS
# =============================================================================

def read_yolo_labels(
    label_path: Path,
) -> List[Tuple[int, float, float, float, float]]:

    objects = []

    if not label_path.exists():
        return objects

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except Exception:
        return objects

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:

            cls_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            objects.append(
                (
                    cls_id,
                    xc,
                    yc,
                    w,
                    h,
                )
            )

        except Exception:
            continue

    return objects


# =============================================================================
# IOU
# =============================================================================

def xywhn_to_xyxy(
    obj,
    width,
    height,
):

    cls_id, xc, yc, w, h = obj

    xc *= width
    yc *= height
    w *= width
    h *= height

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return (
        cls_id,
        x1,
        y1,
        x2,
        y2,
    )


def box_iou(
    box_a,
    box_b,
) -> float:

    _, ax1, ay1, ax2, ay2 = box_a
    _, bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# =============================================================================
# PER-IMAGE F1
# =============================================================================

def calculate_image_f1(
    model: YOLO,
    image_path: Path,
    label_path: Path,
) -> float:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return 0.0

    height, width = image.shape[:2]

    gt_objects = read_yolo_labels(
        label_path
    )

    gt_boxes = [
        xywhn_to_xyxy(
            obj,
            width,
            height,
        )
        for obj in gt_objects
    ]

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        predictions = []

    else:

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            classes = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, cls_id, conf in zip(
                xyxy,
                classes,
                confidences,
            ):

                predictions.append(
                    (
                        int(cls_id),
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                        float(conf),
                    )
                )

    # -------------------------------------------------------------------------
    # Sort predictions by confidence
    # -------------------------------------------------------------------------

    predictions.sort(
        key=lambda x: x[5],
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    # -------------------------------------------------------------------------
    # Greedy class-aware matching
    # -------------------------------------------------------------------------

    for pred in predictions:

        pred_cls = pred[0]

        pred_box = (
            pred_cls,
            pred[1],
            pred[2],
            pred[3],
            pred[4],
        )

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(
            gt_boxes
        ):

            if gt_idx in matched_gt:
                continue

            gt_cls = gt[0]

            if pred_cls != gt_cls:
                continue

            current_iou = box_iou(
                pred_box,
                gt,
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= IOU_THRESHOLD
        ):

            tp += 1

            matched_gt.add(
                best_gt_idx
            )

        else:

            fp += 1

    fn = (
        len(gt_boxes)
        - tp
    )

    denominator = (
        2 * tp
        + fp
        + fn
    )

    if denominator == 0:

        # بدون GT و بدون prediction
        return 1.0

    f1 = (
        2 * tp
        / denominator
    )

    return float(f1)


# =============================================================================
# EVALUATE MODEL ON FINAL TEST
# =============================================================================

def evaluate_model_on_test(
    model_path: Path,
    model_name: str,
) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print(
        f"EVALUATING {model_name}"
    )
    print("=" * 80)

    model = load_model(
        model_path
    )

    test_images_dir = (
        DATASET_ROOT
        / "images"
        / "test"
    )

    test_labels_dir = (
        DATASET_ROOT
        / "labels"
        / "test"
    )

    image_paths = list_images(
        test_images_dir
    )

    records = []

    total = len(
        image_paths
    )

    for idx, image_path in enumerate(
        image_paths,
        start=1,
    ):

        label_path = (
            test_labels_dir
            / f"{image_path.stem}.txt"
        )

        f1 = calculate_image_f1(
            model,
            image_path,
            label_path,
        )

        records.append(
            {
                "image_id": image_path.stem,
                model_name: f1,
            }
        )

        if (
            idx % 100 == 0
            or idx == total
        ):

            print(
                f"[{model_name}] "
                f"{idx}/{total}"
            )

    df = pd.DataFrame(
        records
    )

    output_file = (
        H4_RESULTS_ROOT
        / f"H4_{model_name}_per_image.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved:\n{output_file}"
    )

    return df


# =============================================================================
# MERGE BASE / ADAPTED RESULTS
# =============================================================================

def build_adaptation_comparison(
    base_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
) -> pd.DataFrame:

    merged = pd.merge(
        base_df,
        adapted_df,
        on="image_id",
        how="inner",
    )

    merged = merged.rename(
        columns={
            "YOLOv8_base":
                "non_adapted",
            "YOLOv8_adapted":
                "adapted",
        }
    )

    merged[
        "improvement"
    ] = (
        merged["adapted"]
        - merged["non_adapted"]
    )

    merged[
        "relative_improvement"
    ] = np.where(
        merged["non_adapted"] != 0,
        (
            merged["adapted"]
            - merged["non_adapted"]
        )
        / merged["non_adapted"],
        np.nan,
    )

    output = (
        H4_RESULTS_ROOT
        / "per_image_adaptation.csv"
    )

    merged[
        [
            "image_id",
            "non_adapted",
            "adapted",
            "improvement",
            "relative_improvement",
        ]
    ].to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved comparison:\n{output}"
    )

    return merged


# =============================================================================
# EFFECT SIZE
# =============================================================================

def rank_biserial_from_differences(
    differences: np.ndarray,
) -> float:

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    positive = np.sum(
        differences > 0
    )

    negative = np.sum(
        differences < 0
    )

    return (
        positive - negative
    ) / len(differences)


# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
) -> Tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        SEED
    )

    boot_means = []

    for _ in range(
        n_bootstrap
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means.append(
            np.mean(sample)
        )

    alpha = (
        1
        - confidence
    )

    lower = np.percentile(
        boot_means,
        100 * alpha / 2,
    )

    upper = np.percentile(
        boot_means,
        100 * (
            1 - alpha / 2
        ),
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# H4 STATISTICS
# =============================================================================

def run_h4_statistics(
    df: pd.DataFrame,
) -> Dict:

    print("\n" + "=" * 80)
    print("H4 STATISTICAL ANALYSIS")
    print("=" * 80)

    base = (
        df["non_adapted"]
        .to_numpy(
            dtype=float
        )
    )

    adapted = (
        df["adapted"]
        .to_numpy(
            dtype=float
        )
    )

    improvement = (
        df["improvement"]
        .to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------------------
    # Wilcoxon
    # -------------------------------------------------------------------------

    nonzero = (
        improvement
        != 0
    )

    if np.sum(nonzero) > 0:

        wilcoxon_result = wilcoxon(
            adapted[nonzero],
            base[nonzero],
            alternative="two-sided",
            zero_method="wilcox",
        )

        wilcoxon_stat = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    else:

        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    # -------------------------------------------------------------------------
    # Spearman
    # -------------------------------------------------------------------------

    spearman_result = spearmanr(
        base,
        adapted,
    )

    spearman_rho = float(
        spearman_result.statistic
    )

    spearman_p = float(
        spearman_result.pvalue
    )

    # -------------------------------------------------------------------------
    # Effect size
    # -------------------------------------------------------------------------

    rank_biserial = (
        rank_biserial_from_differences(
            improvement
        )
    )

    # -------------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------------

    ci_low, ci_high = (
        bootstrap_mean_ci(
            improvement
        )
    )

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    summary = {

        "n": int(
            len(df)
        ),

        "non_adapted_mean": float(
            np.mean(base)
        ),

        "adapted_mean": float(
            np.mean(adapted)
        ),

        "non_adapted_median": float(
            np.median(base)
        ),

        "adapted_median": float(
            np.median(adapted)
        ),

        "mean_improvement": float(
            np.mean(improvement)
        ),

        "median_improvement": float(
            np.median(improvement)
        ),

        "std_improvement": float(
            np.std(
                improvement,
                ddof=1,
            )
        ),

        "positive_improvement_count": int(
            np.sum(
                improvement > 0
            )
        ),

        "negative_improvement_count": int(
            np.sum(
                improvement < 0
            )
        ),

        "no_change_count": int(
            np.sum(
                improvement == 0
            )
        ),

        "wilcoxon_statistic": wilcoxon_stat,

        "wilcoxon_p_value": wilcoxon_p,

        "rank_biserial_effect_size":
            float(
                rank_biserial
            ),

        "spearman_rho":
            spearman_rho,

        "spearman_p_value":
            spearman_p,

        "bootstrap_95ci_lower":
            ci_low,

        "bootstrap_95ci_upper":
            ci_high,

        "bootstrap_ci_excludes_zero":
            bool(
                ci_low > 0
                or ci_high < 0
            ),
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        H4_STATS_ROOT
        / "H4_master_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Per-image analysis
    df.to_csv(
        H4_STATS_ROOT
        / "H4_per_image_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        H4_STATS_ROOT
        / "H4_complete_statistics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nMean F1 before adaptation:",
        summary[
            "non_adapted_mean"
        ],
    )

    print(
        "Mean F1 after adaptation:",
        summary[
            "adapted_mean"
        ],
    )

    print(
        "Mean improvement:",
        summary[
            "mean_improvement"
        ],
    )

    print(
        "Wilcoxon p-value:",
        summary[
            "wilcoxon_p_value"
        ],
    )

    print(
        "Rank-biserial effect:",
        summary[
            "rank_biserial_effect_size"
        ],
    )

    print(
        "Bootstrap 95% CI:",
        (
            ci_low,
            ci_high,
        ),
    )

    return summary


# =============================================================================
# FIGURES
# =============================================================================

def create_h4_figures(
    df: pd.DataFrame,
) -> None:

    print("\nCreating H4 figures...")

    base = df[
        "non_adapted"
    ].to_numpy()

    adapted = df[
        "adapted"
    ].to_numpy()

    improvement = df[
        "improvement"
    ].to_numpy()

    # -------------------------------------------------------------------------
    # Figure 1
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [
            base,
            adapted,
        ],
        labels=[
            "Non-adapted",
            "Adapted",
        ],
    )

    plt.ylabel(
        "Per-image F1"
    )

    plt.title(
        "H4: F1 Comparison Before and After Adaptation"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_1_F1_comparison.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        improvement,
        bins=40,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "F1 Improvement"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Distribution of Per-image F1 Improvement"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_2_improvement_distribution.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        base,
        adapted,
        alpha=0.35,
        s=10,
    )

    min_v = min(
        np.min(base),
        np.min(adapted),
    )

    max_v = max(
        np.max(base),
        np.max(adapted),
    )

    plt.plot(
        [
            min_v,
            max_v,
        ],
        [
            min_v,
            max_v,
        ],
        linestyle="--",
    )

    plt.xlabel(
        "Non-adapted F1"
    )

    plt.ylabel(
        "Adapted F1"
    )

    plt.title(
        "Paired F1 Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_3_paired_scatter.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 4
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        improvement
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.ylabel(
        "F1 Improvement"
    )

    plt.title(
        "Distribution of Adaptation Improvement"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_4_boxplot.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 5
    # -------------------------------------------------------------------------

    sorted_improvement = np.sort(
        improvement
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        sorted_improvement
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Sorted Image Index"
    )

    plt.ylabel(
        "F1 Improvement"
    )

    plt.title(
        "Sorted Per-image Improvement"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_5_sorted_improvement.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 6
    # -------------------------------------------------------------------------

    sorted_values = np.sort(
        improvement
    )

    y = (
        np.arange(
            1,
            len(
                sorted_values
            ) + 1,
        )
        / len(
            sorted_values
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        sorted_values,
        y,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "F1 Improvement"
    )

    plt.ylabel(
        "ECDF"
    )

    plt.title(
        "Empirical Cumulative Distribution of Improvement"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_6_ecdf_improvement.png",
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Figure 7
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    if len(df) > 150:

        indices = rng.choice(
            len(df),
            size=150,
            replace=False,
        )

    else:

        indices = np.arange(
            len(df)
        )

    plt.figure(
        figsize=(10, 6)
    )

    for idx in indices:

        plt.plot(
            [
                0,
                1,
            ],
            [
                base[idx],
                adapted[idx],
            ],
            alpha=0.15,
        )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Non-adapted",
            "Adapted",
        ],
    )

    plt.ylabel(
        "Per-image F1"
    )

    plt.title(
        "Paired Per-image F1 Changes"
    )

    plt.tight_layout()

    plt.savefig(
        H4_FIGURES_ROOT
        / "H4_Figure_7_paired_lines.png",
        dpi=300,
    )

    plt.close()

    print(
        f"Figures saved to:\n"
        f"{H4_FIGURES_ROOT}"
    )


# =============================================================================
# INTEGRITY CHECK
# =============================================================================

def create_integrity_check(
    adapted_checkpoint: Path,
) -> None:

    test_images = list_images(
        DATASET_ROOT
        / "images"
        / "test"
    )

    integrity = {

        "experiment": "H4",

        "selected_model": "YOLOv8",

        "base_checkpoint": str(
            BASE_MODEL
        ),

        "adapted_checkpoint": str(
            adapted_checkpoint
        ),

        "image_size": IMAGE_SIZE,

        "confidence_threshold":
            CONF_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "seed": SEED,

        "adaptation_train_images":
            MAX_TRAIN_IMAGES,

        "adaptation_val_images":
            MAX_VAL_IMAGES,

        "adaptation_epochs":
            ADAPT_EPOCHS,

        "adaptation_batch":
            ADAPT_BATCH,

        "adaptation_workers":
            ADAPT_WORKERS,

        "amp":
            ADAPT_AMP,

        "cache":
            ADAPT_CACHE,

        "final_test_images":
            len(test_images),

        "final_test_used_for_adaptation":
            False,

        "adaptation_type":
            "controlled_synthetic_domain_adaptation",

        "real_external_target_dataset":
            False,
    }

    output = (
        H4_RESULTS_ROOT
        / "H4_integrity_check.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            integrity,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nIntegrity report:\n{output}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "H4 END-TO-END SYNTHETIC DOMAIN ADAPTATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nBase model:\n{BASE_MODEL}"
    )

    print(
        f"\nDataset:\n{DATASET_ROOT}"
    )

    print(
        "\nSpeed configuration:"
    )

    print(
        f"  batch   = {ADAPT_BATCH}"
    )

    print(
        f"  workers = {ADAPT_WORKERS}"
    )

    print(
        f"  amp     = {ADAPT_AMP}"
    )

    print(
        f"  cache   = {ADAPT_CACHE}"
    )

    print(
        f"  imgsz   = {IMAGE_SIZE}"
    )

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    set_seed(
        SEED
    )

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------

    create_directories()

    # -------------------------------------------------------------------------
    # Build adaptation dataset
    # -------------------------------------------------------------------------

    build_adaptation_dataset()

    # -------------------------------------------------------------------------
    # Train adapted YOLOv8
    # -------------------------------------------------------------------------

    adapted_checkpoint = (
        train_adapted_model()
    )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    create_integrity_check(
        adapted_checkpoint
    )

    # -------------------------------------------------------------------------
    # Evaluate BASE model
    # -------------------------------------------------------------------------

    base_df_raw = (
        evaluate_model_on_test(
            BASE_MODEL,
            "YOLOv8_base",
        )
    )

    # -------------------------------------------------------------------------
    # Evaluate ADAPTED model
    # -------------------------------------------------------------------------

    adapted_df_raw = (
        evaluate_model_on_test(
            adapted_checkpoint,
            "YOLOv8_adapted",
        )
    )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    comparison_df = (
        build_adaptation_comparison(
            base_df_raw,
            adapted_df_raw,
        )
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    statistics = (
        run_h4_statistics(
            comparison_df
        )
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    create_h4_figures(
        comparison_df
    )

    # -------------------------------------------------------------------------
    # Final configuration
    # -------------------------------------------------------------------------

    final_configuration = {

        "experiment": "H4",

        "selected_model": "YOLOv8",

        "base_checkpoint": str(
            BASE_MODEL
        ),

        "adapted_checkpoint": str(
            adapted_checkpoint
        ),

        "dataset": str(
            DATASET_ROOT
        ),

        "adaptation_dataset": str(
            ADAPT_DATASET_ROOT
        ),

        "imgsz": IMAGE_SIZE,

        "conf": CONF_THRESHOLD,

        "iou": IOU_THRESHOLD,

        "epochs": ADAPT_EPOCHS,

        "batch": ADAPT_BATCH,

        "workers": ADAPT_WORKERS,

        "amp": ADAPT_AMP,

        "cache": ADAPT_CACHE,

        "lr0": ADAPT_LR,

        "optimizer": "AdamW",

        "seed": SEED,

        "patience": ADAPT_PATIENCE,

        "final_test_used_for_adaptation":
            False,

        "adaptation_type":
            "controlled_synthetic_domain_adaptation",

        "statistics": statistics,
    }

    with open(
        H4_RESULTS_ROOT
        / "H4_run_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_configuration,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final message
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "H4 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBase mean F1: "
        f"{statistics['non_adapted_mean']:.6f}"
    )

    print(
        f"Adapted mean F1: "
        f"{statistics['adapted_mean']:.6f}"
    )

    print(
        f"Mean improvement: "
        f"{statistics['mean_improvement']:.6f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{statistics['wilcoxon_p_value']:.6e}"
    )

    print(
        f"Rank-biserial effect: "
        f"{statistics['rank_biserial_effect_size']:.6f}"
    )

    print(
        "\nResults:"
    )

    print(
        H4_RESULTS_ROOT
    )

    print(
        "\nStatistics:"
    )

    print(
        H4_STATS_ROOT
    )

    print(
        "\nFigures:"
    )

    print(
        H4_FIGURES_ROOT
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()