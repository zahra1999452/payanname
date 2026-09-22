from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT / "configs")
)

from config import (
    TARGET_DOMAIN_ROOT,
    TARGET_IMAGES,
    TARGET_LABELS,
)


def main():

    print("=" * 70)
    print("H3 TARGET DOMAIN CHECK")
    print("=" * 70)

    print("Target root  :", TARGET_DOMAIN_ROOT)
    print("Target images:", TARGET_IMAGES)
    print("Target labels:", TARGET_LABELS)

    if not TARGET_DOMAIN_ROOT.exists():
        print()
        print(
            "WARNING: target-domain dataset does not exist yet."
        )
        print(
            "This is a configuration check only."
        )
        return 0

    print()
    print(
        "Target-domain root exists."
    )

    if not TARGET_IMAGES.exists():
        print(
            "WARNING: images directory not found."
        )

    if not TARGET_LABELS.exists():
        print(
            "WARNING: labels directory not found."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())