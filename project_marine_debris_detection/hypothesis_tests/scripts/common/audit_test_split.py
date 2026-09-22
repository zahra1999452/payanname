from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT / "configs")
)

from config import (
    DATASET,
    TEST_IMAGES,
    TEST_LABELS,
    DATA_YAML,
    CLASSES,
)


def main():

    print("=" * 70)
    print("TEST SPLIT AUDIT")
    print("=" * 70)

    print("Dataset:", DATASET)
    print("Images :", TEST_IMAGES)
    print("Labels :", TEST_LABELS)
    print("YAML   :", DATA_YAML)

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(TEST_IMAGES)

    if not TEST_LABELS.exists():
        raise FileNotFoundError(TEST_LABELS)

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }
    )

    labels = sorted(
        p for p in TEST_LABELS.iterdir()
        if p.suffix.lower() == ".txt"
    )

    print()
    print("Number of test images :", len(images))
    print("Number of test labels :", len(labels))
    print("Number of classes     :", len(CLASSES))

    print()
    print("Classes:")

    for i, name in enumerate(CLASSES):
        print(f"{i:02d}  {name}")

    print()
    print("Audit completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())