# -*- coding: utf-8 -*-

"""
================================================================================
H3 Statistical Analysis
Domain-Shift Robustness Comparison
YOLOv8 vs RT-DETR vs RTMDet
================================================================================

ورودی:
    results/H3/per_image_domain_shift.csv

ستون‌های الزامی:
    image_id
    model
    degradation

تفسیر:
    degradation کمتر = پایداری/robustness بیشتر

آزمون‌ها:
    1. Descriptive statistics
    2. Friedman test
    3. Kendall's W
    4. Pairwise Wilcoxon signed-rank
    5. Holm correction
    6. Rank-biserial correlation
    7. Median difference
    8. Direction of effect

خروجی:
    results/statistics/H3/H3_friedman.csv
    results/statistics/H3/H3_pairwise_wilcoxon_holm.csv
    results/statistics/H3/H3_statistics_summary.csv
================================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT
    / "results"
    / "H3"
    / "per_image_domain_shift.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "statistics"
    / "H3"
)

OUTPUT_FRIEDMAN = (
    OUTPUT_DIR
    / "H3_friedman.csv"
)

OUTPUT_PAIRWISE = (
    OUTPUT_DIR
    / "H3_pairwise_wilcoxon_holm.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "H3_statistics_summary.csv"
)


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
# HELPER FUNCTIONS
# =============================================================================

def rank_biserial_correlation(x, y):
    """
    Rank-biserial correlation برای Wilcoxon signed-rank.

    diff = x - y

    مقدار مثبت:
        x > y

    مقدار منفی:
        x < y

    در H3 چون degradation اندازه‌گیری می‌شود:
        مقدار منفی => مدل اول degradation کمتری دارد
                    => robustness بالاتری دارد
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y

    # حذف اختلاف‌های صفر
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)

    # رتبه‌بندی قدرمطلق اختلاف‌ها
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()

    positive_rank_sum = ranks[diff > 0].sum()
    negative_rank_sum = ranks[diff < 0].sum()

    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / denominator


def interpret_effect_size(r):
    """
    تفسیر اندازه اثر rank-biserial correlation.
    """

    absolute_r = abs(r)

    if absolute_r < 0.10:
        return "negligible"

    if absolute_r < 0.30:
        return "small"

    if absolute_r < 0.50:
        return "moderate"

    return "large"


def holm_correction(p_values):
    """
    Holm-Bonferroni correction.

    ورودی:
        list / array of raw p-values

    خروجی:
        corrected p-values
    """

    p_values = np.asarray(p_values, dtype=float)

    m = len(p_values)

    if m == 0:
        return np.array([])

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted = np.empty(m, dtype=float)

    previous = 0.0

    for i, p in enumerate(sorted_p):

        corrected = (m - i) * p

        # corrected p-value نباید از 1 بیشتر شود
        corrected = min(corrected, 1.0)

        # monotonicity
        corrected = max(corrected, previous)

        adjusted[order[i]] = corrected

        previous = corrected

    return adjusted


def significance_label(p):
    """
    تعیین معنی‌داری آماری.
    """

    if pd.isna(p):
        return "NA"

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return "ns"


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_input(df):

    required_columns = [
        "image_id",
        "model",
        "degradation",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # حذف رکوردهای مدل‌های غیرضروری
    df = df[
        df["model"].isin(MODELS)
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows found for the required models."
        )

    # تبدیل degradation به عدد
    df["degradation"] = pd.to_numeric(
        df["degradation"],
        errors="coerce",
    )

    # حذف مقادیر نامعتبر
    before = len(df)

    df = df.dropna(
        subset=[
            "image_id",
            "model",
            "degradation",
        ]
    ).copy()

    after = len(df)

    if before != after:
        print(
            f"WARNING: Removed {before - after} "
            f"rows with invalid values."
        )

    return df


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def calculate_descriptive_statistics(pivot):

    rows = []

    for model in MODELS:

        values = pivot[model].dropna()

        if len(values) == 0:
            continue

        q1 = values.quantile(0.25)
        median = values.median()
        q3 = values.quantile(0.75)

        rows.append({
            "model": model,
            "n": len(values),
            "mean_degradation": values.mean(),
            "std_degradation": values.std(
                ddof=1
            ),
            "median_degradation": median,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": values.min(),
            "max": values.max(),
        })

    return pd.DataFrame(rows)


# =============================================================================
# FRIEDMAN TEST
# =============================================================================

def run_friedman(pivot):

    complete = pivot.dropna(
        subset=MODELS
    ).copy()

    n = len(complete)

    if n < 2:
        raise ValueError(
            "Not enough complete paired observations "
            "for Friedman test."
        )

    statistic, p_value = friedmanchisquare(
        complete["YOLOv8"].to_numpy(),
        complete["RT-DETR"].to_numpy(),
        complete["RTMDet"].to_numpy(),
    )

    k = len(MODELS)

    if n > 0 and k > 1:
        kendall_w = (
            statistic
            / (n * (k - 1))
        )
    else:
        kendall_w = np.nan

    return {
        "test": "Friedman",
        "statistic": statistic,
        "p_value": p_value,
        "n_images": n,
        "n_models": k,
        "kendall_w": kendall_w,
        "significance": significance_label(
            p_value
        ),
    }


# =============================================================================
# PAIRWISE WILCOXON
# =============================================================================

def run_pairwise_tests(pivot):

    results = []

    for model_a, model_b in PAIRS:

        pair = pivot[
            [model_a, model_b]
        ].dropna()

        x = pair[model_a].to_numpy(
            dtype=float
        )

        y = pair[model_b].to_numpy(
            dtype=float
        )

        n = len(pair)

        if n < 2:

            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": n,
                "wilcoxon_statistic": np.nan,
                "p_value_raw": np.nan,
                "p_value_holm": np.nan,
                "median_a": np.nan,
                "median_b": np.nan,
                "median_difference_a_minus_b": np.nan,
                "mean_difference_a_minus_b": np.nan,
                "rank_biserial_r": np.nan,
                "effect_size_interpretation": "NA",
                "direction": "NA",
                "significant": False,
            })

            continue

        difference = x - y

        median_a = np.median(x)
        median_b = np.median(y)

        median_difference = (
            median_a - median_b
        )

        mean_difference = (
            np.mean(x) - np.mean(y)
        )

        # اگر تمام اختلاف‌ها صفر باشند
        if np.allclose(
            difference,
            0,
            atol=1e-15,
        ):

            statistic = 0.0
            p_value = 1.0
            r_rb = 0.0

        else:

            try:

                statistic, p_value = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

            except Exception:

                statistic = np.nan
                p_value = np.nan

            r_rb = rank_biserial_correlation(
                x,
                y,
            )

        # Interpretation
        if median_difference < 0:
            direction = (
                f"{model_a} lower degradation "
                f"(better robustness)"
            )

        elif median_difference > 0:
            direction = (
                f"{model_b} lower degradation "
                f"(better robustness)"
            )

        else:
            direction = "No median difference"

        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_pairs": n,
            "wilcoxon_statistic": statistic,
            "p_value_raw": p_value,
            "p_value_holm": np.nan,
            "median_a": median_a,
            "median_b": median_b,
            "median_difference_a_minus_b": (
                median_difference
            ),
            "mean_difference_a_minus_b": (
                mean_difference
            ),
            "rank_biserial_r": r_rb,
            "effect_size_interpretation":
                interpret_effect_size(r_rb),
            "direction": direction,
            "significant": False,
        })

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Holm correction
    # -------------------------------------------------------------------------

    valid_mask = (
        results_df["p_value_raw"]
        .notna()
    )

    valid_p = (
        results_df.loc[
            valid_mask,
            "p_value_raw",
        ]
        .to_numpy()
    )

    if len(valid_p) > 0:

        corrected = holm_correction(
            valid_p
        )

        results_df.loc[
            valid_mask,
            "p_value_holm",
        ] = corrected

    # -------------------------------------------------------------------------
    # Significance
    # -------------------------------------------------------------------------

    results_df["significant"] = (
        results_df["p_value_holm"]
        < ALPHA
    )

    results_df["significance"] = (
        results_df["p_value_holm"]
        .apply(significance_label)
    )

    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("H3 STATISTICAL ANALYSIS")
    print("Domain-Shift Robustness Comparison")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check input
    # -------------------------------------------------------------------------

    print()
    print("INPUT:")
    print(INPUT)

    if not INPUT.exists():

        raise FileNotFoundError(
            f"\nH3 input file not found:\n{INPUT}\n\n"
            "Expected file:\n"
            "results/H3/per_image_domain_shift.csv"
        )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        INPUT
    )

    print()
    print(
        f"Raw rows: {len(df):,}"
    )

    print(
        "Columns:",
        ", ".join(df.columns),
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    df = validate_input(
        df
    )

    print(
        f"Valid rows: {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Model counts
    # -------------------------------------------------------------------------

    print()
    print("MODEL COUNTS:")

    model_counts = (
        df["model"]
        .value_counts()
        .reindex(MODELS)
        .fillna(0)
        .astype(int)
    )

    for model in MODELS:

        print(
            f"  {model:10s}: "
            f"{model_counts[model]:,}"
        )

    # -------------------------------------------------------------------------
    # Pivot
    # -------------------------------------------------------------------------

    pivot = df.pivot_table(
        index="image_id",
        columns="model",
        values="degradation",
        aggfunc="mean",
    )

    # Ensure all model columns exist
    for model in MODELS:

        if model not in pivot.columns:

            pivot[model] = np.nan

    pivot = pivot[
        MODELS
    ]

    complete = pivot.dropna(
        subset=MODELS
    )

    print()
    print(
        f"Unique images: {len(pivot):,}"
    )

    print(
        f"Complete paired images: "
        f"{len(complete):,}"
    )

    if len(complete) == 0:

        raise ValueError(
            "No complete paired images found."
        )

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    print()
    print(
        "CALCULATING DESCRIPTIVE STATISTICS..."
    )

    descriptive = (
        calculate_descriptive_statistics(
            complete
        )
    )

    # -------------------------------------------------------------------------
    # Friedman
    # -------------------------------------------------------------------------

    print()
    print(
        "CALCULATING FRIEDMAN TEST..."
    )

    friedman_result = run_friedman(
        complete
    )

    friedman_df = pd.DataFrame(
        [friedman_result]
    )

    # -------------------------------------------------------------------------
    # Pairwise
    # -------------------------------------------------------------------------

    print()
    print(
        "CALCULATING PAIRWISE WILCOXON TESTS..."
    )

    pairwise_df = run_pairwise_tests(
        complete
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for _, row in descriptive.iterrows():

        model = row["model"]

        summary_rows.append({
            "section": "descriptive",
            "model": model,
            "n": row["n"],
            "mean_degradation":
                row["mean_degradation"],
            "std_degradation":
                row["std_degradation"],
            "median_degradation":
                row["median_degradation"],
            "q1": row["q1"],
            "q3": row["q3"],
            "iqr": row["iqr"],
            "min": row["min"],
            "max": row["max"],
            "test": "",
            "statistic": np.nan,
            "p_value_raw": np.nan,
            "p_value_holm": np.nan,
            "kendall_w": np.nan,
            "rank_biserial_r": np.nan,
            "effect_size_interpretation": "",
            "direction": "",
            "significant": False,
        })

    # Friedman row
    summary_rows.append({
        "section": "friedman",
        "model": "ALL",
        "n": friedman_result["n_images"],
        "mean_degradation": np.nan,
        "std_degradation": np.nan,
        "median_degradation": np.nan,
        "q1": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "min": np.nan,
        "max": np.nan,
        "test": "Friedman",
        "statistic":
            friedman_result["statistic"],
        "p_value_raw":
            friedman_result["p_value"],
        "p_value_holm": np.nan,
        "kendall_w":
            friedman_result["kendall_w"],
        "rank_biserial_r": np.nan,
        "effect_size_interpretation": "",
        "direction": "",
        "significant":
            friedman_result["p_value"] < ALPHA,
    })

    # Pairwise rows
    for _, row in pairwise_df.iterrows():

        summary_rows.append({
            "section": "pairwise_wilcoxon",
            "model":
                f"{row['model_a']} vs "
                f"{row['model_b']}",
            "n": row["n_pairs"],
            "mean_degradation": np.nan,
            "std_degradation": np.nan,
            "median_degradation": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "test": "Wilcoxon signed-rank",
            "statistic":
                row["wilcoxon_statistic"],
            "p_value_raw":
                row["p_value_raw"],
            "p_value_holm":
                row["p_value_holm"],
            "kendall_w": np.nan,
            "rank_biserial_r":
                row["rank_biserial_r"],
            "effect_size_interpretation":
                row["effect_size_interpretation"],
            "direction":
                row["direction"],
            "significant":
                row["significant"],
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Save Friedman
    # -------------------------------------------------------------------------

    friedman_df.to_csv(
        OUTPUT_FRIEDMAN,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save pairwise
    # -------------------------------------------------------------------------

    pairwise_df.to_csv(
        OUTPUT_PAIRWISE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    print(
        descriptive.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("FRIEDMAN TEST")
    print("=" * 80)

    print(
        friedman_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Friedman p-value: "
        f"{friedman_result['p_value']:.6e}"
    )

    print(
        f"Kendall's W: "
        f"{friedman_result['kendall_w']:.6f}"
    )

    print(
        "Overall significance:",
        "SIGNIFICANT"
        if friedman_result["p_value"] < ALPHA
        else "NOT SIGNIFICANT",
    )

    print()
    print("=" * 80)
    print("PAIRWISE WILCOXON + HOLM")
    print("=" * 80)

    display_columns = [
        "model_a",
        "model_b",
        "n_pairs",
        "wilcoxon_statistic",
        "p_value_raw",
        "p_value_holm",
        "median_difference_a_minus_b",
        "rank_biserial_r",
        "effect_size_interpretation",
        "direction",
        "significant",
    ]

    print(
        pairwise_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        "Friedman:"
    )
    print(
        OUTPUT_FRIEDMAN
    )

    print()
    print(
        "Pairwise:"
    )
    print(
        OUTPUT_PAIRWISE
    )

    print()
    print(
        "Summary:"
    )
    print(
        OUTPUT_SUMMARY
    )

    print()
    print("=" * 80)
    print("H3 ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()