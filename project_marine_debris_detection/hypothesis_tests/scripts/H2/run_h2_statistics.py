# -*- coding: utf-8 -*-

"""
================================================================================
H2 STATISTICAL ANALYSIS
Marine Debris Detection
YOLOv8 vs RT-DETR vs RTMDet

Analysis:
1. Friedman test per corruption
2. Kendall's W effect size
3. Median / IQR degradation
4. Pairwise Wilcoxon signed-rank tests
5. Rank-biserial correlation
6. Holm correction for multiple comparisons
7. Direction of degradation difference
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT
    / "results"
    / "H2"
    / "per_image_degradation.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "statistics"
    / "H2"
)

FRIEDMAN_OUTPUT = OUTPUT_DIR / "H2_friedman.csv"
PAIRWISE_OUTPUT = OUTPUT_DIR / "H2_pairwise_wilcoxon_holm.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "H2_statistics_summary.csv"


# =============================================================================
# SETTINGS
# =============================================================================

MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]

ALPHA = 0.05

PAIRS = [
    ("YOLOv8", "RT-DETR"),
    ("YOLOv8", "RTMDet"),
    ("RT-DETR", "RTMDet"),
]


# =============================================================================
# HELPERS
# =============================================================================

def kendalls_w(friedman_statistic, n, k):
    """
    Kendall's W from Friedman chi-square statistic.

    W = chi2 / [n * (k - 1)]
    """

    if n <= 0 or k <= 1:
        return np.nan

    return friedman_statistic / (n * (k - 1))


def rank_biserial_from_paired(x, y):
    """
    Rank-biserial correlation for paired observations.

    Direction:
        positive -> x tends to be greater than y
        negative -> x tends to be smaller than y

    For degradation:
        negative -> x has lower degradation than y
                     => x is more robust
    """

    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)

    # Remove exact zero differences
    d = d[d != 0]

    n = len(d)

    if n == 0:
        return np.nan

    abs_d = np.abs(d)

    # Rank absolute differences
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()

    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()

    total_rank = n * (n + 1) / 2

    if total_rank == 0:
        return np.nan

    return (w_plus - w_minus) / total_rank


def effect_interpretation(value):
    """
    Conservative interpretation of absolute rank-biserial correlation.
    """

    if pd.isna(value):
        return "undefined"

    a = abs(value)

    if a < 0.10:
        return "negligible"

    if a < 0.30:
        return "small"

    if a < 0.50:
        return "moderate"

    return "large"


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Check input
    # -------------------------------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "image_id",
        "corruption",
        "model",
        "degradation",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Basic cleaning
    # -------------------------------------------------------------------------

    df = df[
        required
    ].copy()

    df["degradation"] = pd.to_numeric(
        df["degradation"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "image_id",
            "corruption",
            "model",
            "degradation",
        ]
    )

    df = df[
        df["model"].isin(MODELS)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid H2 observations remain after cleaning."
        )

    # -------------------------------------------------------------------------
    # Output containers
    # -------------------------------------------------------------------------

    friedman_results = []
    pairwise_results = []

    # -------------------------------------------------------------------------
    # Analyze each corruption separately
    # -------------------------------------------------------------------------

    for corruption, group in df.groupby(
        "corruption",
        sort=True,
    ):

        # ---------------------------------------------------------------------
        # Convert to paired image × model matrix
        # ---------------------------------------------------------------------

        pivot = group.pivot_table(
            index="image_id",
            columns="model",
            values="degradation",
            aggfunc="mean",
        )

        # Keep only images observed for all three models
        pivot = pivot.dropna(
            subset=MODELS
        )

        n = len(pivot)
        k = len(MODELS)

        if n < 3:
            print(
                f"Skipping '{corruption}': "
                f"only {n} complete paired observations."
            )
            continue

        # ---------------------------------------------------------------------
        # Descriptive statistics
        # ---------------------------------------------------------------------

        descriptive = {}

        for model in MODELS:

            values = pivot[model]

            descriptive[
                f"{model}_median"
            ] = values.median()

            descriptive[
                f"{model}_q1"
            ] = values.quantile(0.25)

            descriptive[
                f"{model}_q3"
            ] = values.quantile(0.75)

            descriptive[
                f"{model}_iqr"
            ] = (
                values.quantile(0.75)
                - values.quantile(0.25)
            )

            descriptive[
                f"{model}_mean"
            ] = values.mean()

            descriptive[
                f"{model}_std"
            ] = values.std(ddof=1)

        # ---------------------------------------------------------------------
        # Friedman test
        # ---------------------------------------------------------------------

        statistic, p_value = friedmanchisquare(
            pivot["YOLOv8"],
            pivot["RT-DETR"],
            pivot["RTMDet"],
        )

        W = kendalls_w(
            statistic,
            n,
            k,
        )

        friedman_results.append(
            {
                "corruption": corruption,
                "n_paired_images": n,
                "test": "Friedman",
                "chi_square": statistic,
                "df": k - 1,
                "p_value": p_value,
                "kendalls_W": W,
                "effect_interpretation": (
                    effect_interpretation(W)
                ),
                **descriptive,
            }
        )

        # ---------------------------------------------------------------------
        # Pairwise Wilcoxon tests
        # ---------------------------------------------------------------------

        for model_a, model_b in PAIRS:

            x = pivot[model_a].to_numpy(
                dtype=float
            )

            y = pivot[model_b].to_numpy(
                dtype=float
            )

            differences = x - y

            nonzero = differences[
                differences != 0
            ]

            if len(nonzero) == 0:

                statistic_w = 0.0
                p_pair = 1.0
                rbc = 0.0

            else:

                test = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )

                statistic_w = float(
                    test.statistic
                )

                p_pair = float(
                    test.pvalue
                )

                rbc = rank_biserial_from_paired(
                    x,
                    y,
                )

            median_a = np.median(x)
            median_b = np.median(y)

            median_difference = (
                median_a - median_b
            )

            # Interpretation based on degradation:
            #
            # negative difference:
            # model_a has lower degradation
            # => model_a is more robust
            #
            # positive difference:
            # model_b has lower degradation
            # => model_b is more robust

            if median_difference < 0:
                direction = (
                    f"{model_a} lower degradation"
                )
            elif median_difference > 0:
                direction = (
                    f"{model_b} lower degradation"
                )
            else:
                direction = "equal median degradation"

            pairwise_results.append(
                {
                    "corruption": corruption,
                    "n_paired_images": n,
                    "model_a": model_a,
                    "model_b": model_b,
                    "wilcoxon_statistic": statistic_w,
                    "raw_p": p_pair,
                    "rank_biserial_r": rbc,
                    "effect_interpretation": (
                        effect_interpretation(rbc)
                    ),
                    "median_a": median_a,
                    "median_b": median_b,
                    "median_difference_a_minus_b": (
                        median_difference
                    ),
                    "direction": direction,
                }
            )

    # =========================================================================
    # Convert results to DataFrames
    # =========================================================================

    friedman_df = pd.DataFrame(
        friedman_results
    )

    pairwise_df = pd.DataFrame(
        pairwise_results
    )

    if friedman_df.empty:
        raise RuntimeError(
            "No valid corruption conditions were available "
            "for Friedman analysis."
        )

    if pairwise_df.empty:
        raise RuntimeError(
            "No valid pairwise comparisons were generated."
        )

    # =========================================================================
    # Holm correction
    # =========================================================================

    reject, holm_p, _, _ = multipletests(
        pairwise_df["raw_p"].to_numpy(),
        alpha=ALPHA,
        method="holm",
    )

    pairwise_df[
        "holm_p"
    ] = holm_p

    pairwise_df[
        "significant_alpha_0.05"
    ] = reject

    # =========================================================================
    # Save
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    friedman_df.to_csv(
        FRIEDMAN_OUTPUT,
        index=False,
    )

    pairwise_df.to_csv(
        PAIRWISE_OUTPUT,
        index=False,
    )

    # =========================================================================
    # Overall summary
    # =========================================================================

    summary = []

    for corruption in friedman_df[
        "corruption"
    ].unique():

        f = friedman_df[
            friedman_df["corruption"]
            == corruption
        ].iloc[0]

        pairs = pairwise_df[
            pairwise_df["corruption"]
            == corruption
        ]

        significant_pairs = int(
            pairs[
                "significant_alpha_0.05"
            ].sum()
        )

        summary.append(
            {
                "corruption": corruption,
                "n_paired_images": f[
                    "n_paired_images"
                ],
                "friedman_chi_square": f[
                    "chi_square"
                ],
                "friedman_df": f[
                    "df"
                ],
                "friedman_p": f[
                    "p_value"
                ],
                "kendalls_W": f[
                    "kendalls_W"
                ],
                "significant_pairwise_comparisons": (
                    significant_pairs
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # Console output
    # =========================================================================

    print()
    print("=" * 80)
    print("H2 FRIEDMAN RESULTS")
    print("=" * 80)
    print()

    print(
        friedman_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("H2 PAIRWISE WILCOXON + HOLM")
    print("=" * 80)
    print()

    print(
        pairwise_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("H2 SUMMARY")
    print("=" * 80)
    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        FRIEDMAN_OUTPUT
    )

    print(
        PAIRWISE_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print()
    print("=" * 80)
    print("H2 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()