from pathlib import Path
import cv2
import numpy as np
import sys


PROJECT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT / "configs")
)

from config import (
    TEST_IMAGES,
    CORRUPTION_DATA,
)


def apply_corruption(image, name, severity=2):

    if name == "brightness":
        value = 30 * severity

        result = cv2.convertScaleAbs(
            image,
            alpha=1.0,
            beta=value,
        )

    elif name == "contrast":
        alpha = 1.0 + 0.25 * severity

        result = cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=0,
        )

    elif name == "gaussian_noise":

        sigma = 10 * severity

        noise = np.random.normal(
            0,
            sigma,
            image.shape,
        )

        result = np.clip(
            image.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

    elif name == "gaussian_blur":

        k = 2 * severity + 1

        result = cv2.GaussianBlur(
            image,
            (k, k),
            0,
        )

    elif name == "motion_blur":

        k = 5 * severity

        kernel = np.zeros(
            (k, k),
            dtype=np.float32,
        )

        kernel[k // 2, :] = 1.0 / k

        result = cv2.filter2D(
            image,
            -1,
            kernel,
        )

    elif name == "jpeg_compression":

        quality = max(
            10,
            100 - 15 * severity,
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality,
            ],
        )

        if not ok:
            raise RuntimeError(
                "JPEG encoding failed."
            )

        result = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

    else:
        raise ValueError(
            f"Unknown corruption: {name}"
        )

    return result


def main():

    corruptions = {
        "brightness": 2,
        "contrast": 2,
        "gaussian_noise": 2,
        "gaussian_blur": 2,
        "motion_blur": 2,
        "jpeg_compression": 2,
    }

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )

    for name, severity in corruptions.items():

        output_dir = (
            CORRUPTION_DATA /
            name /
            "images"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_path in images:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            corrupted = apply_corruption(
                image,
                name,
                severity,
            )

            output_path = (
                output_dir /
                image_path.name
            )

            cv2.imwrite(
                str(output_path),
                corrupted,
            )

        print(
            f"Created corruption dataset: {name}"
        )


if __name__ == "__main__":
    main()