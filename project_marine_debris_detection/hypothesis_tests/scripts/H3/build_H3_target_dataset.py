    # -*- coding: utf-8 -*-

"""
===============================================================================
H3 SYNTHETIC / CONTROLLED DOMAIN-SHIFT TARGET DATASET GENERATOR
===============================================================================

Purpose
-------
Build a target-domain dataset from the existing clean TEST split.

The generated target images apply controlled domain-shift effects:

    1. Brightness
    2. Contrast
    3. Color / White Balance
    4. Haze / Fog
    5. Blur
    6. JPEG Compression
    7. Exposure
    8. Saturation
    9. Gaussian Noise
   10. Rotation

Important
---------
Because the target set is generated from the same original test images,
this is a CONTROLLED / SYNTHETIC DOMAIN SHIFT.

It should NOT be described as a real external UAV domain.

Labels:
-------
- Photometric transformations do not change bounding boxes.
- Rotation changes bounding boxes.
- Rotated bounding boxes are recalculated using transformed corners.
- Final boxes are clipped to [0, 1].

Output
------
target/
    images/
        test/
    labels/
        test/
    manifest.csv

The manifest records all parameters used for each image.

Reproducibility
---------------
A deterministic seed is generated for each image from:
    global seed + image filename

Therefore, running the script again produces the same target images
unless --force or another seed is used.
===============================================================================
"""

import os
import cv2
import csv
import math
import shutil
import hashlib
import argparse
from pathlib import Path

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# UTILITY
# =============================================================================

def log(msg):
    print(msg, flush=True)


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_seed(global_seed, image_name):
    """
    Create deterministic per-image seed.

    This avoids Python's randomized hash() behavior.
    """
    text = f"{global_seed}_{image_name}".encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    value = int(digest[:8], 16)

    return value % (2 ** 32)


def list_images(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{image_dir}"
        )

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images.sort(key=lambda x: x.name.lower())

    return images


def corresponding_label(image_path, label_dir):
    return Path(label_dir) / f"{image_path.stem}.txt"


# =============================================================================
# YOLO LABEL FUNCTIONS
# =============================================================================

def load_yolo_labels(label_path):
    """
    Read YOLO labels.

    Format:
        class_id x_center y_center width height

    All coordinates normalized to [0,1].
    """

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                log(
                    f"[WARNING] Invalid label line "
                    f"{label_path.name}:{line_number}"
                )
                continue

            try:
                class_id = int(float(parts[0]))

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                log(
                    f"[WARNING] Non-numeric label "
                    f"{label_path.name}:{line_number}"
                )
                continue

            labels.append(
                {
                    "class_id": class_id,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                }
            )

    return labels


def save_yolo_labels(label_path, labels):
    """
    Save YOLO labels.
    """

    with open(label_path, "w", encoding="utf-8") as f:

        for item in labels:

            class_id = int(item["class_id"])

            xc = np.clip(item["xc"], 0.0, 1.0)
            yc = np.clip(item["yc"], 0.0, 1.0)
            w = np.clip(item["w"], 0.0, 1.0)
            h = np.clip(item["h"], 0.0, 1.0)

            # Skip invalid / zero-area boxes
            if w <= 0 or h <= 0:
                continue

            f.write(
                f"{class_id} "
                f"{xc:.8f} "
                f"{yc:.8f} "
                f"{w:.8f} "
                f"{h:.8f}\n"
            )


# =============================================================================
# YOLO -> CORNERS
# =============================================================================

def yolo_to_corners(label, width, height):
    """
    Convert normalized YOLO bbox to pixel-space corners.

    Returns:
        4x2 array:
        [[x1,y1],
         [x2,y1],
         [x2,y2],
         [x1,y2]]
    """

    xc = label["xc"] * width
    yc = label["yc"] * height
    bw = label["w"] * width
    bh = label["h"] * height

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


# =============================================================================
# ROTATION MATRIX
# =============================================================================

def rotation_matrix_expand(width, height, angle_deg):
    """
    Create an affine matrix for rotation with expanded canvas.

    Returns:
        M
        new_width
        new_height
    """

    center = (width / 2.0, height / 2.0)

    M = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        1.0
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_width = int(
        math.ceil(
            height * sin +
            width * cos
        )
    )

    new_height = int(
        math.ceil(
            height * cos +
            width * sin
        )
    )

    M[0, 2] += (
        new_width / 2.0 -
        center[0]
    )

    M[1, 2] += (
        new_height / 2.0 -
        center[1]
    )

    return M, new_width, new_height


# =============================================================================
# ROTATE IMAGE + LABELS
# =============================================================================

def rotate_image_and_labels(
    image,
    labels,
    angle_deg
):
    """
    Rotate image and transform YOLO bounding boxes.

    Bounding boxes are converted to four corners, transformed by the
    same affine matrix, then converted back to axis-aligned bounding boxes.
    """

    height, width = image.shape[:2]

    M, new_width, new_height = rotation_matrix_expand(
        width,
        height,
        angle_deg
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    transformed_labels = []

    for label in labels:

        corners = yolo_to_corners(
            label,
            width,
            height
        )

        ones = np.ones(
            (corners.shape[0], 1),
            dtype=np.float32
        )

        points_h = np.hstack(
            [corners, ones]
        )

        transformed = points_h @ M.T

        x_min = float(
            np.min(transformed[:, 0])
        )

        y_min = float(
            np.min(transformed[:, 1])
        )

        x_max = float(
            np.max(transformed[:, 0])
        )

        y_max = float(
            np.max(transformed[:, 1])
        )

        # Clip to image
        x_min = np.clip(
            x_min,
            0,
            new_width - 1
        )

        y_min = np.clip(
            y_min,
            0,
            new_height - 1
        )

        x_max = np.clip(
            x_max,
            0,
            new_width - 1
        )

        y_max = np.clip(
            y_max,
            0,
            new_height - 1
        )

        box_width = x_max - x_min
        box_height = y_max - y_min

        if box_width <= 1 or box_height <= 1:
            continue

        xc = (
            (x_min + x_max) / 2.0
        ) / new_width

        yc = (
            (y_min + y_max) / 2.0
        ) / new_height

        bw = box_width / new_width
        bh = box_height / new_height

        transformed_labels.append(
            {
                "class_id": label["class_id"],
                "xc": float(xc),
                "yc": float(yc),
                "w": float(bw),
                "h": float(bh),
            }
        )

    return rotated, transformed_labels


# =============================================================================
# BRIGHTNESS
# =============================================================================

def apply_brightness(image, factor):
    """
    factor:
        1.0 = unchanged
        >1  = brighter
        <1  = darker
    """

    img = image.astype(np.float32)

    img *= factor

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    return img


# =============================================================================
# CONTRAST
# =============================================================================

def apply_contrast(image, factor):
    """
    factor:
        1.0 = unchanged
        >1  = stronger contrast
        <1  = weaker contrast
    """

    img = image.astype(np.float32)

    mean = np.mean(
        img,
        axis=(0, 1),
        keepdims=True
    )

    img = (
        (img - mean) * factor
        + mean
    )

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    return img


# =============================================================================
# EXPOSURE
# =============================================================================

def apply_exposure(image, ev):
    """
    Exposure-like transformation.

    EV:
        +1 -> approximately 2x
        -1 -> approximately 0.5x
    """

    factor = 2.0 ** ev

    img = image.astype(np.float32)

    img *= factor

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    return img


# =============================================================================
# SATURATION
# =============================================================================

def apply_saturation(image, factor):
    """
    factor:
        1.0 = unchanged
        <1  = desaturated
        >1  = more saturated
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 1] *= factor

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255
    )

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


# =============================================================================
# WHITE BALANCE / COLOR SHIFT
# =============================================================================

def apply_white_balance(
    image,
    red_gain,
    green_gain,
    blue_gain
):
    """
    Simple controlled RGB channel gain.

    OpenCV image is BGR.
    """

    img = image.astype(np.float32)

    # B
    img[:, :, 0] *= blue_gain

    # G
    img[:, :, 1] *= green_gain

    # R
    img[:, :, 2] *= red_gain

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    return img


# =============================================================================
# HAZE / FOG
# =============================================================================

def apply_haze(
    image,
    strength
):
    """
    Controlled synthetic haze.

    strength:
        0 = none
        1 = very strong
    """

    h, w = image.shape[:2]

    # Horizontal atmospheric gradient
    y = np.linspace(
        0,
        1,
        h,
        dtype=np.float32
    )

    gradient = np.tile(
        y[:, None],
        (1, w)
    )

    gradient = (
        gradient * 0.5 +
        0.5
    )

    # Airlight color
    airlight = np.array(
        [210, 220, 225],
        dtype=np.float32
    )

    img = image.astype(
        np.float32
    )

    transmission = (
        1.0 -
        strength * gradient[:, :, None]
    )

    transmission = np.clip(
        transmission,
        0.25,
        1.0
    )

    result = (
        img * transmission +
        airlight[None, None, :] *
        (1.0 - transmission)
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result


# =============================================================================
# BLUR
# =============================================================================

def apply_blur(
    image,
    kernel_size
):
    if kernel_size <= 1:
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size
        ),
        sigmaX=0
    )


# =============================================================================
# NOISE
# =============================================================================

def apply_noise(
    image,
    sigma
):
    """
    Add Gaussian noise.
    """

    img = image.astype(
        np.float32
    )

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    ).astype(
        np.float32
    )

    img += noise

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    return img


# =============================================================================
# JPEG COMPRESSION
# =============================================================================

def apply_jpeg_compression(
    image,
    quality
):
    """
    Encode/decode using JPEG to introduce compression artifacts.
    """

    quality = int(
        np.clip(
            quality,
            10,
            100
        )
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ]
    )

    if not success:
        return image

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if decoded is None:
        return image

    return decoded


# =============================================================================
# COMPLETE TARGET TRANSFORMATION
# =============================================================================

def transform_image(
    image,
    labels,
    rng
):
    """
    Apply the complete synthetic domain-shift pipeline.

    The parameter ranges are intentionally moderate so that the generated
    target domain remains visually plausible.
    """

    # -------------------------------------------------------------------------
    # 1. Brightness
    # -------------------------------------------------------------------------

    brightness = float(
        rng.uniform(
            0.78,
            1.22
        )
    )

    image = apply_brightness(
        image,
        brightness
    )

    # -------------------------------------------------------------------------
    # 2. Contrast
    # -------------------------------------------------------------------------

    contrast = float(
        rng.uniform(
            0.80,
            1.20
        )
    )

    image = apply_contrast(
        image,
        contrast
    )

    # -------------------------------------------------------------------------
    # 3. White Balance / Color
    # -------------------------------------------------------------------------

    red_gain = float(
        rng.uniform(
            0.88,
            1.12
        )
    )

    green_gain = float(
        rng.uniform(
            0.93,
            1.07
        )
    )

    blue_gain = float(
        rng.uniform(
            0.88,
            1.12
        )
    )

    image = apply_white_balance(
        image,
        red_gain,
        green_gain,
        blue_gain
    )

    # -------------------------------------------------------------------------
    # 4. Haze / Fog
    # -------------------------------------------------------------------------

    haze_strength = float(
        rng.uniform(
            0.05,
            0.22
        )
    )

    image = apply_haze(
        image,
        haze_strength
    )

    # -------------------------------------------------------------------------
    # 5. Blur
    # -------------------------------------------------------------------------

    blur_kernel = int(
        rng.choice(
            [
                1,
                3,
                3,
                5,
            ]
        )
    )

    image = apply_blur(
        image,
        blur_kernel
    )

    # -------------------------------------------------------------------------
    # 6. Exposure
    # -------------------------------------------------------------------------

    exposure_ev = float(
        rng.uniform(
            -0.45,
            0.45
        )
    )

    image = apply_exposure(
        image,
        exposure_ev
    )

    # -------------------------------------------------------------------------
    # 7. Saturation
    # -------------------------------------------------------------------------

    saturation = float(
        rng.uniform(
            0.78,
            1.22
        )
    )

    image = apply_saturation(
        image,
        saturation
    )

    # -------------------------------------------------------------------------
    # 8. Gaussian Noise
    # -------------------------------------------------------------------------

    noise_sigma = float(
        rng.uniform(
            1.5,
            7.0
        )
    )

    # We temporarily seed NumPy's global RNG for deterministic noise.
    old_state = np.random.get_state()

    try:

        # Generate deterministic seed from RNG
        noise_seed = int(
            rng.integers(
                0,
                2 ** 32 - 1
            )
        )

        np.random.seed(
            noise_seed
        )

        image = apply_noise(
            image,
            noise_sigma
        )

    finally:

        np.random.set_state(
            old_state
        )

    # -------------------------------------------------------------------------
    # 9. Rotation
    # -------------------------------------------------------------------------

    rotation_angle = float(
        rng.uniform(
            -12.0,
            12.0
        )
    )

    image, labels = rotate_image_and_labels(
        image,
        labels,
        rotation_angle
    )

    # -------------------------------------------------------------------------
    # 10. JPEG Compression
    # -------------------------------------------------------------------------

    jpeg_quality = int(
        rng.integers(
            55,
            86
        )
    )

    image = apply_jpeg_compression(
        image,
        jpeg_quality
    )

    parameters = {
        "brightness": brightness,
        "contrast": contrast,
        "red_gain": red_gain,
        "green_gain": green_gain,
        "blue_gain": blue_gain,
        "haze_strength": haze_strength,
        "blur_kernel": blur_kernel,
        "exposure_ev": exposure_ev,
        "saturation": saturation,
        "noise_sigma": noise_sigma,
        "rotation_angle_deg": rotation_angle,
        "jpeg_quality": jpeg_quality,
    }

    return image, labels, parameters


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(
    source_images,
    source_labels,
    target_images,
    target_labels
):
    """
    Verify 1:1 correspondence.
    """

    source_image_ids = {
        p.stem
        for p in source_images
    }

    target_image_ids = {
        p.stem
        for p in target_images
    }

    source_label_ids = {
        p.stem
        for p in source_labels
    }

    target_label_ids = {
        p.stem
        for p in target_labels
    }

    print()
    print("=" * 80)
    print("FINAL DATASET VALIDATION")
    print("=" * 80)

    print(
        f"Source images : {len(source_image_ids)}"
    )

    print(
        f"Target images : {len(target_image_ids)}"
    )

    print(
        f"Source labels : {len(source_label_ids)}"
    )

    print(
        f"Target labels : {len(target_label_ids)}"
    )

    missing_target_images = (
        source_image_ids -
        target_image_ids
    )

    extra_target_images = (
        target_image_ids -
        source_image_ids
    )

    missing_target_labels = (
        source_label_ids -
        target_label_ids
    )

    extra_target_labels = (
        target_label_ids -
        source_label_ids
    )

    if missing_target_images:
        print(
            "[FAIL] Missing target images:",
            len(missing_target_images)
        )

        print(
            list(
                sorted(
                    missing_target_images
                )
            )[:20]
        )

    if extra_target_images:
        print(
            "[FAIL] Extra target images:",
            len(extra_target_images)
        )

    if missing_target_labels:
        print(
            "[FAIL] Missing target labels:",
            len(missing_target_labels)
        )

        print(
            list(
                sorted(
                    missing_target_labels
                )
            )[:20]
        )

    if extra_target_labels:
        print(
            "[FAIL] Extra target labels:",
            len(extra_target_labels)
        )

    success = (
        len(source_image_ids)
        == len(target_image_ids)
        and
        len(missing_target_images)
        == 0
        and
        len(extra_target_images)
        == 0
        and
        len(missing_target_labels)
        == 0
        and
        len(extra_target_labels)
        == 0
    )

    if success:
        print()
        print("[PASS] Source / Target Image-ID validation passed.")
        print("[PASS] Source / Target Label-ID validation passed.")
    else:
        print()
        print("[FAIL] Dataset validation failed.")

    return success


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build H3 controlled synthetic domain-shift target dataset."
        )
    )

    parser.add_argument(
        "--source-images",
        required=True,
        help="Source test image directory."
    )

    parser.add_argument(
        "--source-labels",
        required=True,
        help="Source test label directory."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Target dataset output directory."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target images."
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for testing."
    )

    args = parser.parse_args()

    source_images_dir = Path(
        args.source_images
    )

    source_labels_dir = Path(
        args.source_labels
    )

    output_dir = Path(
        args.output
    )

    target_images_dir = (
        output_dir /
        "images" /
        "test"
    )

    target_labels_dir = (
        output_dir /
        "labels" /
        "test"
    )

    manifest_path = (
        output_dir /
        "manifest.csv"
    )

    # =========================================================================
    # PATH VALIDATION
    # =========================================================================

    if not source_images_dir.exists():
        raise FileNotFoundError(
            f"\nSource image directory does not exist:\n"
            f"{source_images_dir}"
        )

    if not source_labels_dir.exists():
        raise FileNotFoundError(
            f"\nSource label directory does not exist:\n"
            f"{source_labels_dir}"
        )

    # =========================================================================
    # CREATE DIRECTORIES
    # =========================================================================

    make_dir(target_images_dir)
    make_dir(target_labels_dir)

    # =========================================================================
    # SOURCE IMAGES
    # =========================================================================

    source_images = list_images(
        source_images_dir
    )

    if len(source_images) == 0:
        raise RuntimeError(
            "No source images found."
        )

    if args.max_images is not None:

        source_images = source_images[
            :args.max_images
        ]

    log("=" * 80)
    log("H3 SYNTHETIC TARGET DATASET GENERATION")
    log("=" * 80)

    log(
        f"Source images : {source_images_dir}"
    )

    log(
        f"Source labels : {source_labels_dir}"
    )

    log(
        f"Target output : {output_dir}"
    )

    log(
        f"Images to process: {len(source_images)}"
    )

    log(
        f"Global seed: {args.seed}"
    )

    log("=" * 80)

    # =========================================================================
    # MANIFEST
    # =========================================================================

    fieldnames = [
        "image_id",
        "source_image",
        "target_image",
        "source_label",
        "target_label",
        "seed",
        "brightness",
        "contrast",
        "red_gain",
        "green_gain",
        "blue_gain",
        "haze_strength",
        "blur_kernel",
        "exposure_ev",
        "saturation",
        "noise_sigma",
        "rotation_angle_deg",
        "jpeg_quality",
        "source_width",
        "source_height",
        "target_width",
        "target_height",
        "num_source_labels",
        "num_target_labels",
        "status",
    ]

    manifest_rows = []

    # =========================================================================
    # PROCESS
    # =========================================================================

    total = len(source_images)

    for index, source_image_path in enumerate(
        source_images,
        start=1
    ):

        image_id = source_image_path.stem

        target_image_path = (
            target_images_dir /
            source_image_path.name
        )

        source_label_path = (
            source_labels_dir /
            f"{image_id}.txt"
        )

        target_label_path = (
            target_labels_dir /
            f"{image_id}.txt"
        )

        # ---------------------------------------------------------------------
        # Resume support
        # ---------------------------------------------------------------------

        if (
            target_image_path.exists()
            and
            target_label_path.exists()
            and
            not args.force
        ):

            log(
                f"[{index:04d}/{total:04d}] "
                f"[SKIP] {source_image_path.name}"
            )

            # We still need image dimensions for manifest
            existing = cv2.imread(
                str(target_image_path)
            )

            source = cv2.imread(
                str(source_image_path)
            )

            if (
                existing is None
                or source is None
            ):
                log(
                    "[WARNING] Existing output unreadable. "
                    "Reprocessing."
                )
            else:
                source_labels = load_yolo_labels(
                    source_label_path
                )

                target_labels = load_yolo_labels(
                    target_label_path
                )

                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "source_image": source_image_path.name,
                        "target_image": target_image_path.name,
                        "source_label": source_label_path.name,
                        "target_label": target_label_path.name,
                        "seed": stable_seed(
                            args.seed,
                            source_image_path.name
                        ),
                        "brightness": "existing",
                        "contrast": "existing",
                        "red_gain": "existing",
                        "green_gain": "existing",
                        "blue_gain": "existing",
                        "haze_strength": "existing",
                        "blur_kernel": "existing",
                        "exposure_ev": "existing",
                        "saturation": "existing",
                        "noise_sigma": "existing",
                        "rotation_angle_deg": "existing",
                        "jpeg_quality": "existing",
                        "source_width": source.shape[1],
                        "source_height": source.shape[0],
                        "target_width": existing.shape[1],
                        "target_height": existing.shape[0],
                        "num_source_labels": len(source_labels),
                        "num_target_labels": len(target_labels),
                        "status": "existing",
                    }
                )

                continue

        # ---------------------------------------------------------------------
        # Read source image
        # ---------------------------------------------------------------------

        image = cv2.imread(
            str(source_image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            log(
                f"[FAIL] Cannot read image: "
                f"{source_image_path}"
            )

            continue

        source_height, source_width = (
            image.shape[:2]
        )

        # ---------------------------------------------------------------------
        # Read labels
        # ---------------------------------------------------------------------

        labels = load_yolo_labels(
            source_label_path
        )

        source_label_count = len(
            labels
        )

        # ---------------------------------------------------------------------
        # Per-image deterministic RNG
        # ---------------------------------------------------------------------

        image_seed = stable_seed(
            args.seed,
            source_image_path.name
        )

        rng = np.random.default_rng(
            image_seed
        )

        # ---------------------------------------------------------------------
        # Transform
        # ---------------------------------------------------------------------

        try:

            transformed_image, transformed_labels, parameters = (
                transform_image(
                    image.copy(),
                    labels,
                    rng
                )
            )

        except Exception as exc:

            log(
                f"[FAIL] Transformation error "
                f"for {source_image_path.name}: "
                f"{exc}"
            )

            continue

        # ---------------------------------------------------------------------
        # Save target image
        # ---------------------------------------------------------------------

        success = cv2.imwrite(
            str(target_image_path),
            transformed_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ]
        )

        if not success:

            log(
                f"[FAIL] Cannot write target image: "
                f"{target_image_path}"
            )

            continue

        # ---------------------------------------------------------------------
        # Save target labels
        # ---------------------------------------------------------------------

        save_yolo_labels(
            target_label_path,
            transformed_labels
        )

        # ---------------------------------------------------------------------
        # Manifest row
        # ---------------------------------------------------------------------

        target_height, target_width = (
            transformed_image.shape[:2]
        )

        row = {
            "image_id": image_id,
            "source_image": source_image_path.name,
            "target_image": target_image_path.name,
            "source_label": source_label_path.name,
            "target_label": target_label_path.name,
            "seed": image_seed,
            "brightness": parameters["brightness"],
            "contrast": parameters["contrast"],
            "red_gain": parameters["red_gain"],
            "green_gain": parameters["green_gain"],
            "blue_gain": parameters["blue_gain"],
            "haze_strength": parameters["haze_strength"],
            "blur_kernel": parameters["blur_kernel"],
            "exposure_ev": parameters["exposure_ev"],
            "saturation": parameters["saturation"],
            "noise_sigma": parameters["noise_sigma"],
            "rotation_angle_deg": parameters["rotation_angle_deg"],
            "jpeg_quality": parameters["jpeg_quality"],
            "source_width": source_width,
            "source_height": source_height,
            "target_width": target_width,
            "target_height": target_height,
            "num_source_labels": source_label_count,
            "num_target_labels": len(
                transformed_labels
            ),
            "status": "generated",
        }

        manifest_rows.append(row)

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        log(
            f"[{index:04d}/{total:04d}] "
            f"{source_image_path.name} | "
            f"rotation={parameters['rotation_angle_deg']:+.2f}° | "
            f"brightness={parameters['brightness']:.3f} | "
            f"contrast={parameters['contrast']:.3f} | "
            f"haze={parameters['haze_strength']:.3f} | "
            f"blur={parameters['blur_kernel']} | "
            f"noise={parameters['noise_sigma']:.2f} | "
            f"JPEG={parameters['jpeg_quality']}"
        )

    # =========================================================================
    # WRITE MANIFEST
    # =========================================================================

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    target_images = list_images(
        target_images_dir
    )

    target_labels = list(
        target_labels_dir.glob("*.txt")
    )

    source_label_files = list(
        source_labels_dir.glob("*.txt")
    )

    validation_ok = validate_dataset(
        source_images,
        source_label_files,
        target_images,
        target_labels
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 80)
    print("H3 TARGET DATASET GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Source images : {source_images_dir}"
    )

    print(
        f"Source labels : {source_labels_dir}"
    )

    print(
        f"Target images : {target_images_dir}"
    )

    print(
        f"Target labels : {target_labels_dir}"
    )

    print(
        f"Manifest      : {manifest_path}"
    )

    print(
        f"Source count  : {len(source_images)}"
    )

    print(
        f"Target count  : {len(target_images)}"
    )

    print(
        f"Label count   : {len(target_labels)}"
    )

    if validation_ok:
        print()
        print("[SUCCESS] H3 target dataset is ready.")
    else:
        print()
        print("[ERROR] Dataset validation failed.")

    print("=" * 80)


if __name__ == "__main__":
    main()