# -*- coding: utf-8 -*-
"""
====================================================================================================
H4 COMPLETE ANALYSIS
Marine Debris Detection Thesis
====================================================================================================

هدف:
    تحلیل کامل فرضیه چهارم بر اساس مقایسه paired بین:

        non_adapted
        adapted

مدل منتخب:
    YOLOv8

آزمون آماری اصلی:
    Wilcoxon signed-rank test

خروجی‌ها:
    1) آمار توصیفی
    2) Wilcoxon signed-rank
    3) Effect size / rank-biserial correlation
    4) Bootstrap 95% CI
    5) Per-image improvement
    6) CSV های تحلیلی
    7) JSON گزارش
    8) نمودار مقایسه F1
    9) نمودار اختلاف F1
    10) نمودار paired scatter
    11) نمودار distribution
    12) نمودار sorted improvement
    13) summary figure

نکته:
    این اسکریپت adaptation را انجام نمی‌دهد.
    adaptation باید قبلاً انجام شده و فایل per_image_adaptation.csv
    تولید شده باشد.

فرمت حداقل ورودی:

    image_id,non_adapted,adapted
    img_0001.jpg,0.75,0.82
    img_0002.jpg,1.00,1.00
    ...

====================================================================================================
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    wilcoxon,
    rankdata,
    spearmanr,
)

import matplotlib.pyplot as plt


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

PROJECT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = PROJECT / "results" / "H4"
STATISTICS_ROOT = PROJECT / "results" / "statistics" / "H4"
FIGURES_ROOT = PROJECT / "results" / "figures" / "H4"

INPUT = RESULTS_ROOT / "per_image_adaptation.csv"

MODEL_TO_ADAPT = "YOLOv8"

IMAGE_SIZE = 640
SEED = 42

BOOTSTRAP_ITERATIONS = 10000
CONFIDENCE_LEVEL = 0.95

ALPHA = 0.05


# ==================================================================================================
# RANDOM GENERATOR
# ==================================================================================================

RNG = np.random.default_rng(SEED)


# ==================================================================================================
# DIRECTORY SETUP
# ==================================================================================================

def create_directories():

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

def load_data():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"\nH4 input file not found:\n{INPUT}\n\n"
            "ابتدا باید فایل per_image_adaptation.csv تولید شود."
        )

    df = pd.read_csv(INPUT)

    required_columns = [
        "image_id",
        "non_adapted",
        "adapted",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        required_columns
    ].copy()

    # Numeric conversion
    df["non_adapted"] = pd.to_numeric(
        df["non_adapted"],
        errors="coerce",
    )

    df["adapted"] = pd.to_numeric(
        df["adapted"],
        errors="coerce",
    )

    # Remove invalid values
    df = df.dropna(
        subset=[
            "non_adapted",
            "adapted",
        ]
    )

    # Remove duplicated image IDs
    duplicated = df["image_id"].duplicated().sum()

    if duplicated > 0:

        print(
            f"WARNING: {duplicated} duplicated image_id rows found."
        )

        df = (
            df
            .drop_duplicates(
                subset=["image_id"],
                keep="first",
            )
        )

    # F1 validity check
    invalid_non_adapted = (
        (df["non_adapted"] < 0) |
        (df["non_adapted"] > 1)
    ).sum()

    invalid_adapted = (
        (df["adapted"] < 0) |
        (df["adapted"] > 1)
    ).sum()

    if invalid_non_adapted > 0:

        raise ValueError(
            f"Invalid non_adapted F1 values: {invalid_non_adapted}"
        )

    if invalid_adapted > 0:

        raise ValueError(
            f"Invalid adapted F1 values: {invalid_adapted}"
        )

    # Improvement
    df["absolute_improvement"] = (
        df["adapted"] -
        df["non_adapted"]
    )

    # Relative improvement
    df["relative_improvement_percent"] = np.where(
        df["non_adapted"].abs() > 1e-12,
        (
            (
                df["adapted"] -
                df["non_adapted"]
            )
            /
            df["non_adapted"].abs()
        )
        * 100.0,
        np.nan,
    )

    # Absolute difference
    df["absolute_difference"] = (
        df["adapted"] -
        df["non_adapted"]
    ).abs()

    return df


# ==================================================================================================
# DESCRIPTIVE STATISTICS
# ==================================================================================================

def descriptive_statistics(df):

    non_adapted = df["non_adapted"]
    adapted = df["adapted"]

    improvement = df["absolute_improvement"]

    result = {

        "model": MODEL_TO_ADAPT,

        "n_images": int(len(df)),

        "image_size": IMAGE_SIZE,

        "seed": SEED,

        # Non-adapted
        "non_adapted_mean": float(
            non_adapted.mean()
        ),

        "non_adapted_median": float(
            non_adapted.median()
        ),

        "non_adapted_std": float(
            non_adapted.std(ddof=1)
        ),

        "non_adapted_min": float(
            non_adapted.min()
        ),

        "non_adapted_max": float(
            non_adapted.max()
        ),

        # Adapted
        "adapted_mean": float(
            adapted.mean()
        ),

        "adapted_median": float(
            adapted.median()
        ),

        "adapted_std": float(
            adapted.std(ddof=1)
        ),

        "adapted_min": float(
            adapted.min()
        ),

        "adapted_max": float(
            adapted.max()
        ),

        # Improvement
        "mean_improvement": float(
            improvement.mean()
        ),

        "median_improvement": float(
            improvement.median()
        ),

        "std_improvement": float(
            improvement.std(ddof=1)
        ),

        "min_improvement": float(
            improvement.min()
        ),

        "max_improvement": float(
            improvement.max()
        ),

        # Direction
        "n_improved": int(
            (improvement > 0).sum()
        ),

        "n_equal": int(
            (improvement == 0).sum()
        ),

        "n_decreased": int(
            (improvement < 0).sum()
        ),

        "percent_improved": float(
            (improvement > 0).mean() * 100
        ),

        "percent_equal": float(
            (improvement == 0).mean() * 100
        ),

        "percent_decreased": float(
            (improvement < 0).mean() * 100
        ),
    }

    return result


# ==================================================================================================
# WILCOXON
# ==================================================================================================

def run_wilcoxon(df):

    non_adapted = df["non_adapted"].to_numpy()
    adapted = df["adapted"].to_numpy()

    differences = adapted - non_adapted

    non_zero_mask = np.abs(differences) > 1e-12

    n_nonzero = int(
        non_zero_mask.sum()
    )

    if n_nonzero == 0:

        return {
            "test": "Wilcoxon signed-rank",
            "statistic": 0.0,
            "p_value": 1.0,
            "n_total": int(len(df)),
            "n_nonzero_difference": 0,
            "alpha": ALPHA,
            "significant": False,
            "alternative": "two-sided",
        }

    statistic, p_value = wilcoxon(
        non_adapted,
        adapted,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return {

        "test": "Wilcoxon signed-rank",

        "statistic": float(
            statistic
        ),

        "p_value": float(
            p_value
        ),

        "n_total": int(
            len(df)
        ),

        "n_nonzero_difference": n_nonzero,

        "alpha": ALPHA,

        "significant": bool(
            p_value < ALPHA
        ),

        "alternative": "two-sided",
    }


# ==================================================================================================
# RANK-BISERIAL EFFECT SIZE
# ==================================================================================================

def rank_biserial_effect_size(df):

    differences = (
        df["adapted"] -
        df["non_adapted"]
    ).to_numpy()

    differences = differences[
        np.abs(differences) > 1e-12
    ]

    n = len(differences)

    if n == 0:

        return {
            "rank_biserial_r": 0.0,
            "effect_magnitude": "none",
        }

    abs_diff = np.abs(differences)

    ranks = rankdata(
        abs_diff,
        method="average",
    )

    positive_rank_sum = ranks[
        differences > 0
    ].sum()

    negative_rank_sum = ranks[
        differences < 0
    ].sum()

    r_rb = (
        positive_rank_sum -
        negative_rank_sum
    ) / (
        positive_rank_sum +
        negative_rank_sum
    )

    r_rb = float(r_rb)

    magnitude = classify_effect(
        abs(r_rb)
    )

    return {

        "rank_biserial_r": r_rb,

        "effect_magnitude": magnitude,

        "positive_rank_sum": float(
            positive_rank_sum
        ),

        "negative_rank_sum": float(
            negative_rank_sum
        ),

        "n_nonzero": int(n),
    }


# ==================================================================================================
# EFFECT SIZE CLASSIFICATION
# ==================================================================================================

def classify_effect(value):

    if value < 0.10:
        return "negligible"

    if value < 0.30:
        return "small"

    if value < 0.50:
        return "medium"

    return "large"


# ==================================================================================================
# SPEARMAN CORRELATION
# ==================================================================================================

def calculate_spearman(df):

    rho, p = spearmanr(
        df["non_adapted"],
        df["adapted"],
    )

    return {

        "spearman_rho": float(rho),

        "spearman_p_value": float(p),
    }


# ==================================================================================================
# BOOTSTRAP CI
# ==================================================================================================

def bootstrap_mean_difference(
    df,
    iterations=BOOTSTRAP_ITERATIONS,
):

    differences = (
        df["adapted"] -
        df["non_adapted"]
    ).to_numpy()

    n = len(differences)

    if n == 0:

        return {
            "bootstrap_mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    bootstrap_means = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        sample = RNG.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            sample.mean()
        )

    alpha = (
        1 -
        CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return {

        "bootstrap_mean": float(
            differences.mean()
        ),

        "ci_lower": float(
            lower
        ),

        "ci_upper": float(
            upper
        ),

        "confidence_level": CONFIDENCE_LEVEL,

        "iterations": iterations,
    }


# ==================================================================================================
# BOOTSTRAP MEDIAN CI
# ==================================================================================================

def bootstrap_median_difference(
    df,
    iterations=BOOTSTRAP_ITERATIONS,
):

    differences = (
        df["adapted"] -
        df["non_adapted"]
    ).to_numpy()

    n = len(differences)

    bootstrap_medians = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        sample = RNG.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_medians[i] = np.median(
            sample
        )

    alpha = (
        1 -
        CONFIDENCE_LEVEL
    )

    return {

        "bootstrap_median": float(
            np.median(differences)
        ),

        "ci_lower": float(
            np.quantile(
                bootstrap_medians,
                alpha / 2,
            )
        ),

        "ci_upper": float(
            np.quantile(
                bootstrap_medians,
                1 - alpha / 2,
            )
        ),

        "confidence_level": CONFIDENCE_LEVEL,

        "iterations": iterations,
    }


# ==================================================================================================
# SAVE PER-IMAGE RESULTS
# ==================================================================================================

def save_per_image_results(df):

    output = (
        STATISTICS_ROOT /
        "H4_per_image_analysis.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# SAVE DESCRIPTIVE CSV
# ==================================================================================================

def save_descriptive_csv(
    descriptive,
):

    output = (
        STATISTICS_ROOT /
        "H4_descriptive_statistics.csv"
    )

    pd.DataFrame(
        [descriptive]
    ).to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# SAVE WILCOXON CSV
# ==================================================================================================

def save_wilcoxon_csv(
    wilcoxon_result,
):

    output = (
        STATISTICS_ROOT /
        "H4_wilcoxon.csv"
    )

    pd.DataFrame(
        [wilcoxon_result]
    ).to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# SAVE EFFECT SIZE CSV
# ==================================================================================================

def save_effect_csv(
    effect,
):

    output = (
        STATISTICS_ROOT /
        "H4_effect_size.csv"
    )

    pd.DataFrame(
        [effect]
    ).to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# SAVE BOOTSTRAP CSV
# ==================================================================================================

def save_bootstrap_csv(
    mean_bootstrap,
    median_bootstrap,
):

    row = {

        "mean_difference": mean_bootstrap[
            "bootstrap_mean"
        ],

        "mean_ci_lower": mean_bootstrap[
            "ci_lower"
        ],

        "mean_ci_upper": mean_bootstrap[
            "ci_upper"
        ],

        "median_difference": median_bootstrap[
            "bootstrap_median"
        ],

        "median_ci_lower": median_bootstrap[
            "ci_lower"
        ],

        "median_ci_upper": median_bootstrap[
            "ci_upper"
        ],

        "confidence_level": CONFIDENCE_LEVEL,

        "iterations": BOOTSTRAP_ITERATIONS,
    }

    output = (
        STATISTICS_ROOT /
        "H4_bootstrap_CI.csv"
    )

    pd.DataFrame(
        [row]
    ).to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# FIGURE 1 — MEAN / MEDIAN F1
# ==================================================================================================

def plot_f1_comparison(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_1_F1_comparison.png"
    )

    means = [
        df["non_adapted"].mean(),
        df["adapted"].mean(),
    ]

    medians = [
        df["non_adapted"].median(),
        df["adapted"].median(),
    ]

    x = np.arange(2)

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        x - width / 2,
        means,
        width,
        label="Mean F1",
    )

    ax.bar(
        x + width / 2,
        medians,
        width,
        label="Median F1",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Non-adapted",
            "Adapted",
        ]
    )

    ax.set_ylabel(
        "F1-score"
    )

    ax.set_title(
        "H4: F1-score before and after domain adaptation"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 2 — IMPROVEMENT DISTRIBUTION
# ==================================================================================================

def plot_improvement_distribution(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_2_improvement_distribution.png"
    )

    differences = df[
        "absolute_improvement"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.hist(
        differences,
        bins=40,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        differences.mean(),
        linestyle="-",
        linewidth=1.5,
        label=(
            f"Mean = {differences.mean():.4f}"
        ),
    )

    ax.set_xlabel(
        "F1 improvement (adapted − non-adapted)"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.set_title(
        "Distribution of per-image F1 improvement"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 3 — PAIRED SCATTER
# ==================================================================================================

def plot_paired_scatter(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_3_paired_scatter.png"
    )

    x = df[
        "non_adapted"
    ]

    y = df[
        "adapted"
    ]

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        x,
        y,
        alpha=0.25,
        s=18,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="No-change line",
    )

    ax.set_xlabel(
        "Non-adapted F1-score"
    )

    ax.set_ylabel(
        "Adapted F1-score"
    )

    ax.set_title(
        "Paired F1-score comparison"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 4 — BOXPLOT
# ==================================================================================================

def plot_boxplot(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_4_boxplot.png"
    )

    values = [
        df["non_adapted"],
        df["adapted"],
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.boxplot(
        values,
        labels=[
            "Non-adapted",
            "Adapted",
        ],
        showmeans=True,
    )

    ax.set_ylabel(
        "F1-score"
    )

    ax.set_title(
        "Distribution of F1-score before and after adaptation"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 5 — SORTED PER-IMAGE IMPROVEMENT
# ==================================================================================================

def plot_sorted_improvement(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_5_sorted_improvement.png"
    )

    values = np.sort(
        df[
            "absolute_improvement"
        ].to_numpy()
    )

    x = np.arange(
        1,
        len(values) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        x,
        values,
        linewidth=1.2,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Images sorted by F1 improvement"
    )

    ax.set_ylabel(
        "F1 improvement"
    )

    ax.set_title(
        "Sorted per-image improvement after domain adaptation"
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 6 — CUMULATIVE IMPROVEMENT DISTRIBUTION
# ==================================================================================================

def plot_ecdf(
    df,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_6_ecdf_improvement.png"
    )

    values = np.sort(
        df[
            "absolute_improvement"
        ].to_numpy()
    )

    y = (
        np.arange(
            1,
            len(values) + 1,
        )
        /
        len(values)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        values,
        y,
        linewidth=1.5,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "F1 improvement"
    )

    ax.set_ylabel(
        "Cumulative proportion"
    )

    ax.set_title(
        "Empirical cumulative distribution of F1 improvement"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# FIGURE 7 — PAIRED LINES SAMPLE
# ==================================================================================================

def plot_paired_lines(
    df,
    max_images=100,
):

    output = (
        FIGURES_ROOT /
        "H4_Figure_7_paired_lines.png"
    )

    plot_df = df.copy()

    # Sort by non-adapted score
    plot_df = plot_df.sort_values(
        "non_adapted"
    )

    if len(plot_df) > max_images:

        indices = np.linspace(
            0,
            len(plot_df) - 1,
            max_images,
        ).astype(int)

        plot_df = plot_df.iloc[
            indices
        ]

    x = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i in range(
        len(plot_df)
    ):

        ax.plot(
            [
                0,
                1,
            ],
            [
                plot_df.iloc[i][
                    "non_adapted"
                ],
                plot_df.iloc[i][
                    "adapted"
                ],
            ],
            alpha=0.25,
            linewidth=0.8,
        )

    ax.scatter(
        np.zeros(
            len(plot_df)
        ),
        plot_df[
            "non_adapted"
        ],
        s=15,
    )

    ax.scatter(
        np.ones(
            len(plot_df)
        ),
        plot_df[
            "adapted"
        ],
        s=15,
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "Non-adapted",
            "Adapted",
        ]
    )

    ax.set_ylabel(
        "F1-score"
    )

    ax.set_title(
        f"Paired F1-score changes ({len(plot_df)} representative images)"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ==================================================================================================
# SAVE SUMMARY JSON
# ==================================================================================================

def save_json_summary(
    descriptive,
    wilcoxon_result,
    effect,
    spearman,
    mean_bootstrap,
    median_bootstrap,
):

    summary = {

        "experiment": "H4",

        "model": MODEL_TO_ADAPT,

        "image_size": IMAGE_SIZE,

        "seed": SEED,

        "input": str(INPUT),

        "descriptive_statistics": descriptive,

        "wilcoxon": wilcoxon_result,

        "effect_size": effect,

        "spearman": spearman,

        "bootstrap_mean_difference": mean_bootstrap,

        "bootstrap_median_difference": median_bootstrap,

        "interpretation": {

            "significance": (
                "statistically significant"
                if wilcoxon_result[
                    "significant"
                ]
                else
                "not statistically significant"
            ),

            "direction": (
                "improvement"
                if descriptive[
                    "mean_improvement"
                ] > 0
                else
                "decrease"
                if descriptive[
                    "mean_improvement"
                ] < 0
                else
                "no mean change"
            ),

            "effect_magnitude": effect[
                "effect_magnitude"
            ],
        },
    }

    output = (
        STATISTICS_ROOT /
        "H4_complete_statistics.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return output


# ==================================================================================================
# SAVE MASTER SUMMARY CSV
# ==================================================================================================

def save_master_summary(
    descriptive,
    wilcoxon_result,
    effect,
    spearman,
    mean_bootstrap,
    median_bootstrap,
):

    row = {

        **descriptive,

        "wilcoxon_statistic":
            wilcoxon_result[
                "statistic"
            ],

        "wilcoxon_p_value":
            wilcoxon_result[
                "p_value"
            ],

        "wilcoxon_significant":
            wilcoxon_result[
                "significant"
            ],

        "rank_biserial_r":
            effect[
                "rank_biserial_r"
            ],

        "effect_magnitude":
            effect[
                "effect_magnitude"
            ],

        "spearman_rho":
            spearman[
                "spearman_rho"
            ],

        "spearman_p_value":
            spearman[
                "spearman_p_value"
            ],

        "bootstrap_mean_ci_lower":
            mean_bootstrap[
                "ci_lower"
            ],

        "bootstrap_mean_ci_upper":
            mean_bootstrap[
                "ci_upper"
            ],

        "bootstrap_median_ci_lower":
            median_bootstrap[
                "ci_lower"
            ],

        "bootstrap_median_ci_upper":
            median_bootstrap[
                "ci_upper"
            ],
    }

    output = (
        STATISTICS_ROOT /
        "H4_master_summary.csv"
    )

    pd.DataFrame(
        [row]
    ).to_csv(
        output,
        index=False,
    )

    return output


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print()
    print("=" * 100)
    print("H4 COMPLETE ANALYSIS")
    print("=" * 100)

    print()
    print("Model:", MODEL_TO_ADAPT)
    print("Input:", INPUT)
    print("Image size:", IMAGE_SIZE)
    print("Seed:", SEED)

    create_directories()

    # ----------------------------------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------------------------------

    print()
    print("[1/10] Loading data...")

    df = load_data()

    print(
        f"Number of paired images: {len(df):,}"
    )

    # ----------------------------------------------------------------------------------------------
    # Descriptive
    # ----------------------------------------------------------------------------------------------

    print()
    print("[2/10] Descriptive statistics...")

    descriptive = descriptive_statistics(
        df
    )

    # ----------------------------------------------------------------------------------------------
    # Wilcoxon
    # ----------------------------------------------------------------------------------------------

    print()
    print("[3/10] Wilcoxon signed-rank test...")

    wilcoxon_result = run_wilcoxon(
        df
    )

    print(
        f"Statistic: {wilcoxon_result['statistic']:.6f}"
    )

    print(
        f"p-value: {wilcoxon_result['p_value']:.12g}"
    )

    # ----------------------------------------------------------------------------------------------
    # Effect size
    # ----------------------------------------------------------------------------------------------

    print()
    print("[4/10] Effect size...")

    effect = rank_biserial_effect_size(
        df
    )

    print(
        f"Rank-biserial r: "
        f"{effect['rank_biserial_r']:.6f}"
    )

    print(
        f"Magnitude: "
        f"{effect['effect_magnitude']}"
    )

    # ----------------------------------------------------------------------------------------------
    # Spearman
    # ----------------------------------------------------------------------------------------------

    print()
    print("[5/10] Spearman correlation...")

    spearman = calculate_spearman(
        df
    )

    print(
        f"rho: "
        f"{spearman['spearman_rho']:.6f}"
    )

    print(
        f"p-value: "
        f"{spearman['spearman_p_value']:.12g}"
    )

    # ----------------------------------------------------------------------------------------------
    # Bootstrap mean
    # ----------------------------------------------------------------------------------------------

    print()
    print("[6/10] Bootstrap CI for mean improvement...")

    mean_bootstrap = bootstrap_mean_difference(
        df
    )

    print(
        f"Mean improvement: "
        f"{mean_bootstrap['bootstrap_mean']:.6f}"
    )

    print(
        f"95% CI: "
        f"[{mean_bootstrap['ci_lower']:.6f}, "
        f"{mean_bootstrap['ci_upper']:.6f}]"
    )

    # ----------------------------------------------------------------------------------------------
    # Bootstrap median
    # ----------------------------------------------------------------------------------------------

    print()
    print("[7/10] Bootstrap CI for median improvement...")

    median_bootstrap = bootstrap_median_difference(
        df
    )

    print(
        f"Median improvement: "
        f"{median_bootstrap['bootstrap_median']:.6f}"
    )

    print(
        f"95% CI: "
        f"[{median_bootstrap['ci_lower']:.6f}, "
        f"{median_bootstrap['ci_upper']:.6f}]"
    )

    # ----------------------------------------------------------------------------------------------
    # Save statistics
    # ----------------------------------------------------------------------------------------------

    print()
    print("[8/10] Saving statistical results...")

    save_per_image_results(
        df
    )

    save_descriptive_csv(
        descriptive
    )

    save_wilcoxon_csv(
        wilcoxon_result
    )

    save_effect_csv(
        effect
    )

    save_bootstrap_csv(
        mean_bootstrap,
        median_bootstrap,
    )

    save_master_summary(
        descriptive,
        wilcoxon_result,
        effect,
        spearman,
        mean_bootstrap,
        median_bootstrap,
    )

    save_json_summary(
        descriptive,
        wilcoxon_result,
        effect,
        spearman,
        mean_bootstrap,
        median_bootstrap,
    )

    # ----------------------------------------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------------------------------------

    print()
    print("[9/10] Generating figures...")

    figure_paths = []

    figure_paths.append(
        plot_f1_comparison(
            df
        )
    )

    figure_paths.append(
        plot_improvement_distribution(
            df
        )
    )

    figure_paths.append(
        plot_paired_scatter(
            df
        )
    )

    figure_paths.append(
        plot_boxplot(
            df
        )
    )

    figure_paths.append(
        plot_sorted_improvement(
            df
        )
    )

    figure_paths.append(
        plot_ecdf(
            df
        )
    )

    figure_paths.append(
        plot_paired_lines(
            df
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------------------------------------

    print()
    print("[10/10] Final summary...")

    print()
    print("-" * 100)

    print(
        f"Model:                  {MODEL_TO_ADAPT}"
    )

    print(
        f"N paired images:        {len(df):,}"
    )

    print(
        f"Non-adapted mean F1:    "
        f"{descriptive['non_adapted_mean']:.6f}"
    )

    print(
        f"Adapted mean F1:        "
        f"{descriptive['adapted_mean']:.6f}"
    )

    print(
        f"Mean improvement:       "
        f"{descriptive['mean_improvement']:.6f}"
    )

    print(
        f"Median improvement:     "
        f"{descriptive['median_improvement']:.6f}"
    )

    print(
        f"Improved images:        "
        f"{descriptive['n_improved']:,} "
        f"({descriptive['percent_improved']:.2f}%)"
    )

    print(
        f"Equal images:           "
        f"{descriptive['n_equal']:,} "
        f"({descriptive['percent_equal']:.2f}%)"
    )

    print(
        f"Decreased images:       "
        f"{descriptive['n_decreased']:,} "
        f"({descriptive['percent_decreased']:.2f}%)"
    )

    print(
        f"Wilcoxon statistic:     "
        f"{wilcoxon_result['statistic']:.6f}"
    )

    print(
        f"Wilcoxon p-value:       "
        f"{wilcoxon_result['p_value']:.12g}"
    )

    print(
        f"Rank-biserial r:        "
        f"{effect['rank_biserial_r']:.6f}"
    )

    print(
        f"Effect magnitude:       "
        f"{effect['effect_magnitude']}"
    )

    print(
        f"Bootstrap 95% CI mean:  "
        f"[{mean_bootstrap['ci_lower']:.6f}, "
        f"{mean_bootstrap['ci_upper']:.6f}]"
    )

    print(
        f"Statistically significant: "
        f"{wilcoxon_result['significant']}"
    )

    print("-" * 100)

    print()
    print("STATISTICS:")
    print(STATISTICS_ROOT)

    print()
    print("FIGURES:")

    for path in figure_paths:

        print(path)

    print()
    print("=" * 100)
    print("H4 ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print()


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()