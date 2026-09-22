from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT /
    "results" /
    "H4" /
    "per_image_adaptation.csv"
)

OUTPUT = (
    PROJECT /
    "results" /
    "statistics" /
    "H4_statistics.csv"
)


def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = [
        "image_id",
        "non_adapted",
        "adapted",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}"
            )

    df = df.dropna(
        subset=[
            "non_adapted",
            "adapted",
        ]
    )

    statistic, p = wilcoxon(
        df["non_adapted"],
        df["adapted"],
        zero_method="wilcox",
        alternative="two-sided",
    )

    result = pd.DataFrame(
        [
            {
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "p_value": p,
                "n_images": len(df),
            }
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print(result.to_string(index=False))
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()