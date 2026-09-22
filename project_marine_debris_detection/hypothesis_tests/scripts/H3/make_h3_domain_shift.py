
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd


# ================================================================================================
# CONFIGURATION
# ================================================================================================

DEFAULT_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

GEOMETRIC_CONDITIONS = {
    "rotation_minus_15",
    "rotation_plus_15",
    "perspective_direction",
    "shear_direction",
    "scale_view_shift",
    "uav_coastal_domain_shift",
}

PHOTOMETRIC_CONDITIONS = {
    "low_light_extreme",
    "high_light_extreme",
    "gamma_dark",
    "gamma_bright",
    "color_temperature_cool",
    "color_temperature_warm",
}


@dataclass
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float


@dataclass
class TransformResult:
    image: np.ndarray
    boxes: List[Box]
    transform_type: str
    seed: int


# ================================================================================================
# GENERAL UTILITIES
# ================================================================================================

def fail(message: str, code: int = 1) -> None:
    print("\n" + "=" * 100)
    print("FATAL ERROR")
    print("=" * 100)
    print(message)
    print("=" * 100)
    raise SystemExit(code)


def stable_seed(global_seed: int, image_name: str, condition: str) -> int:
    """
    Stable seed independent of Python's randomized hash().
    """
    raw = f"{global_seed}|{image_name}|{condition}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:8], 16)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> List[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def image_key(path: Path) -> str:
    return path.stem


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    ensure_dir(path.parent)

    ext = path.suffix.lower()
    params = []

    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


# ================================================================================================
# YOLO LABEL IO
# ================================================================================================

def load_yolo_labels(label_path: Path) -> List[Box]:
    """
    Read normalized YOLO:
        class xc yc w h
    """
    boxes: List[Box] = []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return boxes

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}, line {line_no}: {line}"
            )

        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            raise ValueError(
                f"Non-numeric YOLO label at {label_path}, line {line_no}: {line}"
            )

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(
                f"Out-of-range YOLO bbox at {label_path}, line {line_no}: {line}"
            )

        boxes.append(Box(cls, xc, yc, w, h))

    return boxes


def save_yolo_labels(label_path: Path, boxes: List[Box]) -> None:
    ensure_dir(label_path.parent)

    lines = []
    for b in boxes:
        lines.append(
            f"{b.cls} "
            f"{b.xc:.8f} "
            f"{b.yc:.8f} "
            f"{b.w:.8f} "
            f"{b.h:.8f}"
        )

    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8"
    )


def box_to_pixels(box: Box, width: int, height: int) -> np.ndarray:
    """
    Return four bbox corners:
        [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    xc = box.xc * width
    yc = box.yc * height
    bw = box.w * width
    bh = box.h * height

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def pixels_to_box(
    cls: int,
    corners: np.ndarray,
    width: int,
    height: int,
    min_area_fraction: float = 0.05,
) -> Optional[Box]:
    """
    Convert transformed corners to an axis-aligned YOLO bbox.

    min_area_fraction:
        Reject boxes if the visible clipped bbox area becomes too small
        relative to the original transformed bbox area.
    """
    if corners.shape != (4, 2):
        raise ValueError("Expected four transformed bbox corners.")

    x_min = float(np.min(corners[:, 0]))
    y_min = float(np.min(corners[:, 1]))
    x_max = float(np.max(corners[:, 0]))
    y_max = float(np.max(corners[:, 1]))

    # Original transformed rectangular envelope area.
    original_area = max(0.0, (x_max - x_min) * (y_max - y_min))

    # Clip to image boundaries.
    x1 = float(np.clip(x_min, 0, width - 1))
    y1 = float(np.clip(y_min, 0, height - 1))
    x2 = float(np.clip(x_max, 0, width - 1))
    y2 = float(np.clip(y_max, 0, height - 1))

    clipped_w = max(0.0, x2 - x1)
    clipped_h = max(0.0, y2 - y1)
    clipped_area = clipped_w * clipped_h

    if clipped_area <= 1.0:
        return None

    if original_area > 1.0:
        visible_fraction = clipped_area / original_area
        if visible_fraction < min_area_fraction:
            return None

    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = clipped_w / width
    bh = clipped_h / height

    # Final numerical safety.
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    bw = float(np.clip(bw, 1e-8, 1.0))
    bh = float(np.clip(bh, 1e-8, 1.0))

    return Box(cls, xc, yc, bw, bh)


def transform_boxes(
    boxes: List[Box],
    matrix: np.ndarray,
    width: int,
    height: int,
    min_visible_fraction: float = 0.05,
) -> List[Box]:
    """
    Apply a 3x3 affine/projective transformation to bbox corners.
    """
    transformed_boxes: List[Box] = []

    for box in boxes:
        corners = box_to_pixels(box, width, height)

        pts = corners.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)

        new_box = pixels_to_box(
            box.cls,
            warped,
            width,
            height,
            min_area_fraction=min_visible_fraction,
        )

        if new_box is not None:
            transformed_boxes.append(new_box)

    return transformed_boxes


# ================================================================================================
# PHOTOMETRIC DOMAIN SHIFTS
# ================================================================================================

def adjust_brightness_contrast(
    image: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    dst = alpha * src + beta
    """
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out


def gamma_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    gamma < 1 -> brighter
    gamma > 1 -> darker
    """
    inv = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv) * 255 for i in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def color_temperature_shift(
    image: np.ndarray,
    blue_gain: float,
    red_gain: float,
    green_gain: float = 1.0,
) -> np.ndarray:
    """
    BGR gains approximate a controlled white-balance/domain shift.
    """
    img = image.astype(np.float32)

    img[:, :, 0] *= blue_gain
    img[:, :, 1] *= green_gain
    img[:, :, 2] *= red_gain

    return np.clip(img, 0, 255).astype(np.uint8)


def add_spatial_illumination_gradient(
    image: np.ndarray,
    strength: float,
    direction: str,
) -> np.ndarray:
    """
    Simulates non-uniform illumination / vignetting.
    No geometry is changed, therefore boxes remain valid.
    """
    h, w = image.shape[:2]

    if direction == "left":
        x = np.linspace(1.0 + strength, 1.0 - strength, w)
        field = np.tile(x[None, :], (h, 1))
    elif direction == "right":
        x = np.linspace(1.0 - strength, 1.0 + strength, w)
        field = np.tile(x[None, :], (h, 1))
    elif direction == "top":
        y = np.linspace(1.0 + strength, 1.0 - strength, h)
        field = np.tile(y[:, None], (1, w))
    else:
        y = np.linspace(1.0 - strength, 1.0 + strength, h)
        field = np.tile(y[:, None], (1, w))

    out = image.astype(np.float32) * field[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ================================================================================================
# GEOMETRIC DOMAIN SHIFTS
# ================================================================================================

def rotation_matrix(
    width: int,
    height: int,
    angle: float,
    scale: float = 1.0,
) -> np.ndarray:
    center = (width / 2.0, height / 2.0)
    return cv2.getRotationMatrix2D(center, angle, scale)


def affine_to_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :] = matrix_2x3
    return matrix


def warp_affine_same_size(
    image: np.ndarray,
    matrix_2x3: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]

    return cv2.warpAffine(
        image,
        matrix_2x3,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def make_perspective_matrix(
    width: int,
    height: int,
    rng: np.random.Generator,
    strength: float = 0.075,
) -> np.ndarray:
    """
    Moderate perspective distortion.
    The canvas remains unchanged.
    """
    margin_x = width * strength
    margin_y = height * strength

    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    jitter = np.array([
        [rng.uniform(-margin_x, margin_x), rng.uniform(-margin_y, margin_y)],
        [rng.uniform(-margin_x, margin_x), rng.uniform(-margin_y, margin_y)],
        [rng.uniform(-margin_x, margin_x), rng.uniform(-margin_y, margin_y)],
        [rng.uniform(-margin_x, margin_x), rng.uniform(-margin_y, margin_y)],
    ], dtype=np.float32)

    dst = src + jitter

    return cv2.getPerspectiveTransform(src, dst)


def shear_matrix(
    width: int,
    height: int,
    shear_x: float,
    shear_y: float,
) -> np.ndarray:
    """
    Shear around image center while preserving canvas size.
    """
    cx = width / 2.0
    cy = height / 2.0

    matrix = np.array([
        [1.0, shear_x, -shear_x * cy],
        [shear_y, 1.0, -shear_y * cx],
    ], dtype=np.float32)

    # Correct translation to center transformation approximately.
    matrix[0, 2] += (1.0 - 1.0) * cx
    matrix[1, 2] += (1.0 - 1.0) * cy

    return matrix


def scale_view_matrix(
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    """
    Simulates apparent camera distance / object scale.
    Uses reflected borders rather than cropping.
    """
    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [scale, 0.0, (1.0 - scale) * cx],
        [0.0, scale, (1.0 - scale) * cy],
    ], dtype=np.float32)


# ================================================================================================
# CONDITION PIPELINES
# ================================================================================================

def apply_condition(
    image: np.ndarray,
    boxes: List[Box],
    condition: str,
    seed: int,
    min_visible_fraction: float,
) -> TransformResult:

    rng = np.random.default_rng(seed)

    h, w = image.shape[:2]

    # --------------------------------------------------------------------------------------------
    # PURE PHOTOMETRIC CONDITIONS
    # --------------------------------------------------------------------------------------------

    if condition == "low_light_extreme":
        out = adjust_brightness_contrast(image, alpha=0.62, beta=-38)
        out = gamma_transform(out, gamma=1.35)
        out = add_spatial_illumination_gradient(
            out, strength=0.28, direction="right"
        )
        return TransformResult(out, boxes, "photometric", seed)

    if condition == "high_light_extreme":
        out = adjust_brightness_contrast(image, alpha=1.12, beta=48)
        out = gamma_transform(out, gamma=0.72)
        out = add_spatial_illumination_gradient(
            out, strength=0.22, direction="left"
        )
        return TransformResult(out, boxes, "photometric", seed)

    if condition == "gamma_dark":
        out = gamma_transform(image, gamma=1.55)
        return TransformResult(out, boxes, "photometric", seed)

    if condition == "gamma_bright":
        out = gamma_transform(image, gamma=0.62)
        return TransformResult(out, boxes, "photometric", seed)

    if condition == "color_temperature_cool":
        out = color_temperature_shift(
            image,
            blue_gain=1.22,
            green_gain=1.02,
            red_gain=0.78,
        )
        out = adjust_brightness_contrast(out, alpha=0.96, beta=-8)
        return TransformResult(out, boxes, "photometric", seed)

    if condition == "color_temperature_warm":
        out = color_temperature_shift(
            image,
            blue_gain=0.78,
            green_gain=1.02,
            red_gain=1.22,
        )
        out = adjust_brightness_contrast(out, alpha=0.96, beta=5)
        return TransformResult(out, boxes, "photometric", seed)

    # --------------------------------------------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------------------------------------------

    if condition == "rotation_minus_15":
        m2 = rotation_matrix(w, h, angle=-15.0, scale=1.0)
        m3 = affine_to_homogeneous(m2)

        out = warp_affine_same_size(image, m2)
        new_boxes = transform_boxes(
            boxes, m3, w, h, min_visible_fraction
        )

        return TransformResult(out, new_boxes, "affine_rotation", seed)

    if condition == "rotation_plus_15":
        m2 = rotation_matrix(w, h, angle=15.0, scale=1.0)
        m3 = affine_to_homogeneous(m2)

        out = warp_affine_same_size(image, m2)
        new_boxes = transform_boxes(
            boxes, m3, w, h, min_visible_fraction
        )

        return TransformResult(out, new_boxes, "affine_rotation", seed)

    # --------------------------------------------------------------------------------------------
    # PERSPECTIVE
    # --------------------------------------------------------------------------------------------

    if condition == "perspective_direction":
        m3 = make_perspective_matrix(
            w, h, rng, strength=0.065
        )

        out = cv2.warpPerspective(
            image,
            m3,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        new_boxes = transform_boxes(
            boxes, m3, w, h, min_visible_fraction
        )

        return TransformResult(out, new_boxes, "projective_perspective", seed)

    # --------------------------------------------------------------------------------------------
    # SHEAR
    # --------------------------------------------------------------------------------------------

    if condition == "shear_direction":
        # Moderate directional shear.
        sx = float(rng.choice([-0.075, 0.075]))
        sy = float(rng.choice([-0.025, 0.025]))

        m2 = shear_matrix(w, h, sx, sy)
        m3 = affine_to_homogeneous(m2)

        out = warp_affine_same_size(image, m2)
        new_boxes = transform_boxes(
            boxes, m3, w, h, min_visible_fraction
        )

        return TransformResult(out, new_boxes, "affine_shear", seed)

    # --------------------------------------------------------------------------------------------
    # SCALE / VIEW-DISTANCE SHIFT
    # --------------------------------------------------------------------------------------------

    if condition == "scale_view_shift":
        scale = float(rng.choice([0.78, 1.22]))

        m2 = scale_view_matrix(w, h, scale)
        m3 = affine_to_homogeneous(m2)

        out = warp_affine_same_size(image, m2)
        new_boxes = transform_boxes(
            boxes, m3, w, h, min_visible_fraction
        )

        return TransformResult(out, new_boxes, "affine_scale", seed)

    # --------------------------------------------------------------------------------------------
    # COMBINED DOMAIN SHIFT
    # --------------------------------------------------------------------------------------------

    if condition == "uav_coastal_domain_shift":
        # Stage 1: strong but controlled illumination shift.
        out = adjust_brightness_contrast(
            image,
            alpha=0.72,
            beta=-30,
        )
        out = gamma_transform(out, gamma=1.28)

        # Directional illumination gradient.
        out = add_spatial_illumination_gradient(
            out,
            strength=0.24,
            direction="right",
        )

        # White-balance / atmospheric color shift.
        out = color_temperature_shift(
            out,
            blue_gain=1.12,
            green_gain=1.02,
            red_gain=0.88,
        )

        # Stage 2: camera/viewpoint change.
        # Rotation + shear + mild perspective.
        angle = float(rng.choice([-12.0, 12.0]))

        rot2 = rotation_matrix(w, h, angle=angle, scale=1.0)
        rot3 = affine_to_homogeneous(rot2)

        out = warp_affine_same_size(out, rot2)
        boxes2 = transform_boxes(
            boxes,
            rot3,
            w,
            h,
            min_visible_fraction,
        )

        # Mild directional shear.
        sx = float(rng.choice([-0.045, 0.045]))
        sy = float(rng.choice([-0.015, 0.015]))
        shear2 = shear_matrix(w, h, sx, sy)
        shear3 = affine_to_homogeneous(shear2)

        out = warp_affine_same_size(out, shear2)
        boxes3 = transform_boxes(
            boxes2,
            shear3,
            w,
            h,
            min_visible_fraction,
        )

        # Final perspective perturbation.
        persp = make_perspective_matrix(
            w, h, rng, strength=0.045
        )

        out = cv2.warpPerspective(
            out,
            persp,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        boxes4 = transform_boxes(
            boxes3,
            persp,
            w,
            h,
            min_visible_fraction,
        )

        return TransformResult(
            out,
            boxes4,
            "combined_photometric_affine_projective",
            seed,
        )

    raise ValueError(f"Unknown H3 condition: {condition}")


# ================================================================================================
# VALIDATION
# ================================================================================================

def validate_boxes(boxes: List[Box]) -> Tuple[bool, str]:
    for i, b in enumerate(boxes):
        if b.cls < 0:
            return False, f"negative class at index {i}"

        values = [b.xc, b.yc, b.w, b.h]

        if not all(np.isfinite(v) for v in values):
            return False, f"non-finite bbox at index {i}"

        if not (0.0 <= b.xc <= 1.0):
            return False, f"xc out of range at index {i}: {b.xc}"

        if not (0.0 <= b.yc <= 1.0):
            return False, f"yc out of range at index {i}: {b.yc}"

        if not (0.0 < b.w <= 1.0):
            return False, f"w out of range at index {i}: {b.w}"

        if not (0.0 < b.h <= 1.0):
            return False, f"h out of range at index {i}: {b.h}"

    return True, "OK"


def verify_source_pairing(
    image_paths: List[Path],
    label_root: Path,
) -> Dict[str, int]:
    missing = []
    empty = 0

    for img in image_paths:
        label = label_root / f"{img.stem}.txt"

        if not label.exists():
            missing.append(img.name)
            continue

        if label.read_text(encoding="utf-8").strip() == "":
            empty += 1

    if missing:
        preview = ", ".join(missing[:10])
        fail(
            f"Missing {len(missing)} source label files. "
            f"Examples: {preview}"
        )

    return {
        "images": len(image_paths),
        "labels": len(image_paths) - empty,
        "empty_label_files": empty,
    }


def verify_generated_pairing(
    image_root: Path,
    label_root: Path,
) -> Dict[str, int]:
    images = list_images(image_root)

    missing_labels = []
    for img in images:
        label = label_root / f"{img.stem}.txt"
        if not label.exists():
            missing_labels.append(img.name)

    orphan_labels = []
    for label in label_root.glob("*.txt"):
        img_candidates = [
            image_root / f"{label.stem}.jpg",
            image_root / f"{label.stem}.jpeg",
            image_root / f"{label.stem}.png",
            image_root / f"{label.stem}.bmp",
            image_root / f"{label.stem}.webp",
        ]
        if not any(p.exists() for p in img_candidates):
            orphan_labels.append(label.name)

    return {
        "images": len(images),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
    }


# ================================================================================================
# GENERATION
# ================================================================================================

def generate_condition(
    source_images: Path,
    source_labels: Path,
    output_root: Path,
    condition: str,
    seed: int,
    min_visible_fraction: float,
    jpeg_quality: int,
) -> Dict:

    condition_root = output_root / condition
    image_root = condition_root / "images"
    label_root = condition_root / "labels"

    ensure_dir(image_root)
    ensure_dir(label_root)

    images = list_images(source_images)

    records = []

    for idx, image_path in enumerate(images, start=1):
        label_path = source_labels / f"{image_path.stem}.txt"

        image = load_image(image_path)
        boxes = load_yolo_labels(label_path)

        h, w = image.shape[:2]

        local_seed = stable_seed(
            seed,
            image_path.name,
            condition,
        )

        result = apply_condition(
            image=image,
            boxes=boxes,
            condition=condition,
            seed=local_seed,
            min_visible_fraction=min_visible_fraction,
        )

        valid, reason = validate_boxes(result.boxes)
        if not valid:
            raise RuntimeError(
                f"Generated invalid annotation for "
                f"{image_path.name} / {condition}: {reason}"
            )

        # Preserve original image extension where possible.
        output_image = image_root / image_path.name
        output_label = label_root / f"{image_path.stem}.txt"

        save_image(
            output_image,
            result.image,
            quality=jpeg_quality,
        )
        save_yolo_labels(output_label, result.boxes)

        records.append({
            "source_image": str(image_path),
            "source_label": str(label_path),
            "target_image": str(output_image),
            "target_label": str(output_label),
            "image_id": image_path.stem,
            "condition": condition,
            "transform_type": result.transform_type,
            "seed": local_seed,
            "width": w,
            "height": h,
            "source_num_boxes": len(boxes),
            "target_num_boxes": len(result.boxes),
            "boxes_removed": len(boxes) - len(result.boxes),
        })

        if idx % 250 == 0 or idx == len(images):
            print(
                f"  [{idx:5d}/{len(images)}] "
                f"{condition:30s}"
            )

    manifest_path = condition_root / "manifest.csv"
    pd.DataFrame(records).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    pairing = verify_generated_pairing(
        image_root,
        label_root,
    )

    summary = {
        "condition": condition,
        "transform_type": records[0]["transform_type"] if records else None,
        "num_images": len(records),
        "num_source_boxes": int(
            sum(r["source_num_boxes"] for r in records)
        ),
        "num_target_boxes": int(
            sum(r["target_num_boxes"] for r in records)
        ),
        "num_removed_boxes": int(
            sum(r["boxes_removed"] for r in records)
        ),
        "mean_boxes_removed_per_image": float(
            np.mean([r["boxes_removed"] for r in records])
        ) if records else 0.0,
        "pairing_check": pairing,
        "seed": seed,
        "min_visible_fraction": min_visible_fraction,
        "jpeg_quality": jpeg_quality,
    }

    (condition_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate H3 simulated domain-shift dataset with correct YOLO bbox transforms."
    )

    parser.add_argument(
        "--source-images",
        required=True,
        help="Clean source TEST images directory.",
    )

    parser.add_argument(
        "--source-labels",
        required=True,
        help="Clean source TEST YOLO labels directory.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root for H3 domain-shift dataset.",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        choices=CONDITIONS,
        help="H3 conditions to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.05,
        help="Minimum retained bbox area fraction after geometric transforms.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=98,
        help="JPEG output quality. H3 is not intended as JPEG corruption.",
    )

    args = parser.parse_args()

    source_images = Path(args.source_images)
    source_labels = Path(args.source_labels)
    output_root = Path(args.output_root)

    print("=" * 100)
    print("H3 DOMAIN-SHIFT DATASET GENERATOR")
    print("=" * 100)
    print(f"Source images       : {source_images}")
    print(f"Source labels       : {source_labels}")
    print(f"Output root         : {output_root}")
    print(f"Seed                : {args.seed}")
    print(f"Conditions          : {len(args.conditions)}")
    print(f"Min visible bbox    : {args.min_visible_fraction}")
    print("=" * 100)

    # --------------------------------------------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------------------------------------------

    if not source_images.exists():
        fail(f"Source images directory does not exist:\n{source_images}")

    if not source_labels.exists():
        fail(f"Source labels directory does not exist:\n{source_labels}")

    if not source_images.is_dir():
        fail(f"Source images path is not a directory:\n{source_images}")

    if not source_labels.is_dir():
        fail(f"Source labels path is not a directory:\n{source_labels}")

    if not (0.0 < args.min_visible_fraction <= 1.0):
        fail("--min-visible-fraction must be > 0 and <= 1.")

    if not (50 <= args.jpeg_quality <= 100):
        fail("--jpeg-quality must be between 50 and 100.")

    images = list_images(source_images)

    if not images:
        fail(f"No supported images found in:\n{source_images}")

    # --------------------------------------------------------------------------------------------
    # SOURCE INTEGRITY
    # --------------------------------------------------------------------------------------------

    source_check = verify_source_pairing(
        images,
        source_labels,
    )

    print("\nSOURCE INTEGRITY")
    print("-" * 100)
    print(json.dumps(source_check, indent=2))

    # --------------------------------------------------------------------------------------------
    # OUTPUT ROOT
    # --------------------------------------------------------------------------------------------

    ensure_dir(output_root)

    all_summaries = []

    # --------------------------------------------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------------------------------------------

    print("\nGENERATION")
    print("-" * 100)

    for condition in args.conditions:
        print(f"\n>>> Generating: {condition}")

        summary = generate_condition(
            source_images=source_images,
            source_labels=source_labels,
            output_root=output_root,
            condition=condition,
            seed=args.seed,
            min_visible_fraction=args.min_visible_fraction,
            jpeg_quality=args.jpeg_quality,
        )

        all_summaries.append(summary)

        print(
            f"    images={summary['num_images']} | "
            f"source_boxes={summary['num_source_boxes']} | "
            f"target_boxes={summary['num_target_boxes']} | "
            f"removed={summary['num_removed_boxes']}"
        )

    # --------------------------------------------------------------------------------------------
    # GLOBAL MANIFEST
    # --------------------------------------------------------------------------------------------

    global_manifest_rows = []

    for condition in args.conditions:
        manifest = output_root / condition / "manifest.csv"

        if manifest.exists():
            df = pd.read_csv(manifest)
            global_manifest_rows.append(df)

    if global_manifest_rows:
        global_manifest = pd.concat(
            global_manifest_rows,
            ignore_index=True,
        )
        global_manifest.to_csv(
            output_root / "H3_global_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------------------------------------------

    global_summary = {
        "experiment": "H3_simulated_domain_shift",
        "source_images": str(source_images),
        "source_labels": str(source_labels),
        "output_root": str(output_root),
        "seed": args.seed,
        "conditions": args.conditions,
        "num_source_images": len(images),
        "source_integrity": source_check,
        "condition_summaries": all_summaries,
        "methodological_note": (
            "This dataset represents controlled/simulated domain shift. "
            "It must not be described as external real-world target-domain "
            "generalization."
        ),
        "h2_separation_note": (
            "The H3 conditions emphasize illumination, color temperature, "
            "orientation, affine geometry, perspective, scale/viewpoint, "
            "and their controlled combination rather than the H2 corruption "
            "set of brightness, contrast, Gaussian noise, Gaussian blur, "
            "motion blur, and JPEG compression."
        ),
    }

    summary_path = output_root / "H3_dataset_summary.json"
    summary_path.write_text(
        json.dumps(
            global_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("H3 DATASET GENERATION COMPLETE")
    print("=" * 100)

    for s in all_summaries:
        print(
            f"{s['condition']:30s} | "
            f"images={s['num_images']:5d} | "
            f"boxes={s['num_target_boxes']:7d} | "
            f"removed={s['num_removed_boxes']:6d}"
        )

    print("\nOutput:")
    print(f"  {output_root}")
    print(f"  {summary_path}")
    print(f"  {output_root / 'H3_global_manifest.csv'}")

    print("\nIMPORTANT:")
    print(
        "H3 is a simulated domain-shift experiment. "
        "Do not describe it as evaluation on a real external target domain."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()