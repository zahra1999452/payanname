from pathlib import Path
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[2]

INPUT = PROJECT / "results" / "H1" / "per_image_metrics.csv"

OUTPUT = PROJECT / "results" / "statistics" / "H1_statistics.csv"


MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "image_id",
        "model",
        "f1",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Required column missing: {col}"
            )

    pivot = df.pivot_table(
        index="image_id",
        columns="model",
        values="f1",
        aggfunc="mean",
    )

    pivot = pivot.dropna(
        subset=MODELS
    )

    if len(pivot) < 3:
        raise ValueError(
            "Insufficient matched observations."
        )

    stat, p = friedmanchisquare(
        pivot["YOLOv8"],
        pivot["RT-DETR"],
        pivot["RTMDet"],
    )

    pairs = [
        ("YOLOv8", "RT-DETR"),
        ("YOLOv8", "RTMDet"),
        ("RT-DETR", "RTMDet"),
    ]

    raw_p = []
    rows = []

    for a, b in pairs:

        w, wp = wilcoxon(
            pivot[a],
            pivot[b],
            zero_method="wilcox",
            alternative="two-sided",
        )

        raw_p.append(wp)

        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "wilcoxon_statistic": w,
                "raw_p": wp,
            }
        )

    reject, corrected_p, _, _ = multipletests(
        raw_p,
        method="holm",
    )

    for row, cp, r in zip(
        rows,
        corrected_p,
        reject,
    ):
        row["holm_p"] = cp
        row["significant_alpha_0.05"] = bool(r)

    output = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("Friedman statistic:", stat)
    print("Friedman p-value :", p)
    print()
    print(output.to_string(index=False))
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()