
"""
====================================================================================================
H4 MODEL SELECTION
====================================================================================================

انتخاب مدل مناسب برای H4 بر اساس نتایج H1، H2 و H3.

منطق انتخاب:
----------------------------------------------------------------------------------------------------
H1:
    معیار = F1
    بیشتر بودن F1 بهتر است.

H2:
    معیار = degradation
    کمتر بودن degradation بهتر است.
    degradation مثبت  -> افت عملکرد
    degradation صفر    -> بدون افت
    degradation منفی   -> بهبود عملکرد

H3:
    معیار = degradation
    کمتر بودن degradation بهتر است.

برای H3 دو نوع رتبه‌بندی محاسبه می‌شود:

1) Overall H3:
   میانگین degradation تمام رکوردها.

2) Condition-balanced H3:
   ابتدا برای هر condition میانگین degradation هر مدل محاسبه می‌شود،
   سپس مدل‌ها داخل هر condition رتبه‌بندی می‌شوند.
   در این روش همه شرایط H3 وزن یکسان دارند.

انتخاب نهایی:
----------------------------------------------------------------------------------------------------
اولویت 1:
    تعداد رتبه‌های اول بیشتر

اولویت 2:
    مجموع رتبه‌ها کمتر

اولویت 3:
    میانگین رتبه کمتر

اولویت 4:
    H3 condition-balanced بهتر

خروجی‌ها:
----------------------------------------------------------------------------------------------------
results/H4/H4_model_selection_summary.csv
results/H4/H4_model_ranking.csv
results/H4/H4_h3_condition_ranking.csv
results/H4/H4_h3_condition_wins.csv
results/H4/H4_selected_model.txt
results/H4/H4_selection_details.json

====================================================================================================
"""

from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np


# ==================================================================================================
# PATHS
# ==================================================================================================

PROJECT = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection\hypothesis_tests"
)

RESULTS = PROJECT / "results"

H1_FILE = RESULTS / "H1" / "per_image_metrics.csv"
H2_FILE = RESULTS / "H2" / "per_image_degradation.csv"
H3_FILE = RESULTS / "H3" / "H3_all_conditions_per_image_domain_shift.csv"

H4_DIR = RESULTS / "H4"

H4_SUMMARY = H4_DIR / "H4_model_selection_summary.csv"
H4_RANKING = H4_DIR / "H4_model_ranking.csv"
H4_H3_CONDITION_RANKING = H4_DIR / "H4_h3_condition_ranking.csv"
H4_H3_CONDITION_WINS = H4_DIR / "H4_h3_condition_wins.csv"
H4_SELECTED = H4_DIR / "H4_selected_model.txt"
H4_DETAILS = H4_DIR / "H4_selection_details.json"


# ==================================================================================================
# MODELS
# ==================================================================================================

MODELS = [
    "YOLOv8",
    "RT-DETR",
    "RTMDet",
]


# ==================================================================================================
# UTILITY FUNCTIONS
# ==================================================================================================

def print_header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def normalize_model_name(value):
    """
    یکسان‌سازی نام مدل‌ها.
    """

    if pd.isna(value):
        return None

    x = str(value).strip().lower()

    mapping = {
        "yolov8": "YOLOv8",
        "yolo": "YOLOv8",
        "yolov8n": "YOLOv8",

        "rt-detr": "RT-DETR",
        "rtdetr": "RT-DETR",
        "rt detr": "RT-DETR",
        "rtdetr-l": "RT-DETR",
        "rtdetr_l": "RT-DETR",

        "rtmdet": "RTMDet",
        "rtm-det": "RTMDet",
        "rtm det": "RTMDet",
        "rtmdet-tiny": "RTMDet",
        "rtmdet_tiny": "RTMDet",
    }

    return mapping.get(x, str(value).strip())


def validate_file(path, name):
    """
    بررسی وجود فایل.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} پیدا نشد:\n{path}\n"
        )

    print(f"[OK] {name}")
    print(f"     {path}")


def check_columns(df, required, name):
    """
    بررسی ستون‌های ضروری.
    """

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nستون‌های زیر در {name} وجود ندارند:\n"
            f"{missing}\n\n"
            f"ستون‌های موجود:\n"
            f"{list(df.columns)}"
        )


def clean_model_column(df, column="model"):
    """
    پاک‌سازی نام مدل‌ها.
    """

    df = df.copy()
    df[column] = df[column].apply(normalize_model_name)

    df = df[df[column].isin(MODELS)].copy()

    return df


# ==================================================================================================
# H1
# ==================================================================================================

def process_h1():
    """
    H1:
        F1 بیشتر = بهتر
    """

    print_header("H1 ANALYSIS")

    df = pd.read_csv(H1_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    check_columns(
        df,
        ["model", "f1"],
        "H1 per_image_metrics.csv"
    )

    df = clean_model_column(df)

    df["f1"] = pd.to_numeric(
        df["f1"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["model", "f1"]
    )

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            n_images=("f1", "count"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            std_f1=("f1", "std"),
        )
    )

    summary = summary[
        summary["model"].isin(MODELS)
    ].copy()

    # رتبه:
    # F1 بیشتر = رتبه بهتر
    summary = summary.sort_values(
        by="mean_f1",
        ascending=False
    ).reset_index(drop=True)

    summary["H1_rank"] = np.arange(
        1,
        len(summary) + 1
    )

    print()
    print("H1 MODEL PERFORMANCE")
    print(
        summary[
            [
                "model",
                "n_images",
                "mean_f1",
                "median_f1",
                "std_f1",
                "H1_rank",
            ]
        ].to_string(index=False)
    )

    return summary


# ==================================================================================================
# H2
# ==================================================================================================

def process_h2():
    """
    H2:
        degradation کمتر = بهتر
    """

    print_header("H2 ANALYSIS")

    df = pd.read_csv(H2_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    check_columns(
        df,
        ["model", "degradation"],
        "H2 per_image_degradation.csv"
    )

    df = clean_model_column(df)

    df["degradation"] = pd.to_numeric(
        df["degradation"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["model", "degradation"]
    )

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            n_observations=("degradation", "count"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
            std_degradation=("degradation", "std"),
        )
    )

    summary = summary[
        summary["model"].isin(MODELS)
    ].copy()

    # degradation کمتر = بهتر
    summary = summary.sort_values(
        by="mean_degradation",
        ascending=True
    ).reset_index(drop=True)

    summary["H2_rank"] = np.arange(
        1,
        len(summary) + 1
    )

    print()
    print("H2 MODEL ROBUSTNESS")
    print(
        summary[
            [
                "model",
                "n_observations",
                "mean_degradation",
                "median_degradation",
                "std_degradation",
                "H2_rank",
            ]
        ].to_string(index=False)
    )

    return summary


# ==================================================================================================
# H3 OVERALL
# ==================================================================================================

def process_h3():
    """
    H3 overall:
        degradation کمتر = بهتر
    """

    print_header("H3 OVERALL ANALYSIS")

    df = pd.read_csv(H3_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    check_columns(
        df,
        ["model", "degradation"],
        "H3_all_conditions_per_image_domain_shift.csv"
    )

    df = clean_model_column(df)

    df["degradation"] = pd.to_numeric(
        df["degradation"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["model", "degradation"]
    )

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            n_observations=("degradation", "count"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
            std_degradation=("degradation", "std"),
        )
    )

    summary = summary[
        summary["model"].isin(MODELS)
    ].copy()

    summary = summary.sort_values(
        by="mean_degradation",
        ascending=True
    ).reset_index(drop=True)

    summary["H3_rank"] = np.arange(
        1,
        len(summary) + 1
    )

    print()
    print("H3 OVERALL ROBUSTNESS")
    print(
        summary[
            [
                "model",
                "n_observations",
                "mean_degradation",
                "median_degradation",
                "std_degradation",
                "H3_rank",
            ]
        ].to_string(index=False)
    )

    return df, summary


# ==================================================================================================
# H3 CONDITION-BALANCED
# ==================================================================================================

def process_h3_conditions(df):
    """
    برای جلوگیری از وزن‌دهی بیش از حد به شرایطی که داده بیشتری دارند،
    هر condition به صورت مستقل رتبه‌بندی می‌شود.

    condition:
        هر کدام از 12 شرایط H3

    داخل هر condition:
        degradation کمتر = رتبه بهتر
    """

    print_header("H3 CONDITION-BALANCED ANALYSIS")

    if "condition" not in df.columns:
        print(
            "[WARNING] ستون condition در H3 وجود ندارد."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    condition_summary = (
        df.groupby(
            ["condition", "model"],
            as_index=False
        )
        .agg(
            n_observations=("degradation", "count"),
            mean_degradation=("degradation", "mean"),
            median_degradation=("degradation", "median"),
        )
    )

    # رتبه داخل هر condition
    condition_summary["H3_condition_rank"] = (
        condition_summary
        .groupby("condition")["mean_degradation"]
        .rank(
            method="min",
            ascending=True
        )
    )

    condition_summary = condition_summary.sort_values(
        ["condition", "H3_condition_rank", "model"]
    ).reset_index(drop=True)

    # ذخیره جزئیات
    condition_summary.to_csv(
        H4_H3_CONDITION_RANKING,
        index=False,
        encoding="utf-8-sig"
    )

    # شمارش بردها
    wins = (
        condition_summary[
            condition_summary["H3_condition_rank"] == 1
        ]
        .groupby("model")
        .size()
        .reindex(MODELS, fill_value=0)
        .reset_index(name="H3_condition_wins")
    )

    # میانگین رتبه در شرایط
    mean_rank = (
        condition_summary
        .groupby("model")["H3_condition_rank"]
        .mean()
        .reindex(MODELS)
        .reset_index(name="H3_condition_mean_rank")
    )

    wins = wins.merge(
        mean_rank,
        on="model",
        how="left"
    )

    wins.to_csv(
        H4_H3_CONDITION_WINS,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("H3 CONDITION-LEVEL RESULTS")
    print(
        wins.to_string(index=False)
    )

    print()
    print(
        "Condition-level ranking:"
    )

    for condition in sorted(
        condition_summary["condition"].unique()
    ):

        sub = condition_summary[
            condition_summary["condition"] == condition
        ].copy()

        print()
        print(f"  {condition}")

        print(
            sub[
                [
                    "model",
                    "mean_degradation",
                    "H3_condition_rank",
                ]
            ].to_string(index=False)
        )

    return condition_summary, wins


# ==================================================================================================
# COMBINE RANKS
# ==================================================================================================

def combine_results(
    h1_summary,
    h2_summary,
    h3_summary,
    h3_condition_wins
):

    print_header("FINAL MODEL SELECTION")

    h1 = h1_summary[
        ["model", "H1_rank", "mean_f1"]
    ].copy()

    h2 = h2_summary[
        ["model", "H2_rank", "mean_degradation"]
    ].copy()

    h3 = h3_summary[
        ["model", "H3_rank"]
    ].copy()

    ranking = pd.DataFrame({
        "model": MODELS
    })

    ranking = ranking.merge(
        h1,
        on="model",
        how="left"
    )

    ranking = ranking.merge(
        h2,
        on="model",
        how="left"
    )

    ranking = ranking.merge(
        h3,
        on="model",
        how="left"
    )

    # ----------------------------------------------------------------------------------------------
    # تعداد رتبه‌های اول
    # ----------------------------------------------------------------------------------------------

    ranking["n_first_place"] = (
        (ranking["H1_rank"] == 1).astype(int)
        +
        (ranking["H2_rank"] == 1).astype(int)
        +
        (ranking["H3_rank"] == 1).astype(int)
    )

    # ----------------------------------------------------------------------------------------------
    # مجموع رتبه‌ها
    # ----------------------------------------------------------------------------------------------

    ranking["rank_sum"] = (
        ranking["H1_rank"]
        +
        ranking["H2_rank"]
        +
        ranking["H3_rank"]
    )

    ranking["mean_rank"] = (
        ranking["rank_sum"] / 3.0
    )

    # ----------------------------------------------------------------------------------------------
    # H3 condition-balanced
    # ----------------------------------------------------------------------------------------------

    if not h3_condition_wins.empty:

        ranking = ranking.merge(
            h3_condition_wins,
            on="model",
            how="left"
        )

    else:

        ranking["H3_condition_wins"] = 0
        ranking["H3_condition_mean_rank"] = np.nan

    ranking["H3_condition_wins"] = (
        ranking["H3_condition_wins"]
        .fillna(0)
        .astype(int)
    )

    # ----------------------------------------------------------------------------------------------
    # ترتیب نهایی
    # ----------------------------------------------------------------------------------------------

    ranking = ranking.sort_values(
        by=[
            "n_first_place",
            "rank_sum",
            "mean_rank",
            "H3_condition_wins",
            "H3_condition_mean_rank",
        ],
        ascending=[
            False,
            True,
            True,
            False,
            True,
        ]
    ).reset_index(drop=True)

    ranking["final_rank"] = np.arange(
        1,
        len(ranking) + 1
    )

    # ----------------------------------------------------------------------------------------------
    # Selected model
    # ----------------------------------------------------------------------------------------------

    selected_model = ranking.iloc[0]["model"]

    # ----------------------------------------------------------------------------------------------
    # آیا یک مدل در هر سه H1/H2/H3 رتبه اول است؟
    # ----------------------------------------------------------------------------------------------

    triple_winners = ranking[
        ranking["n_first_place"] == 3
    ]["model"].tolist()

    unanimous = len(triple_winners) == 1

    # ----------------------------------------------------------------------------------------------
    # Print
    # ----------------------------------------------------------------------------------------------

    print()
    print("FINAL RANKING")
    print(
        ranking[
            [
                "final_rank",
                "model",
                "H1_rank",
                "H2_rank",
                "H3_rank",
                "n_first_place",
                "rank_sum",
                "mean_rank",
                "H3_condition_wins",
                "H3_condition_mean_rank",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 100)

    if unanimous:

        print(
            f"[STRONG RESULT] {selected_model} رتبه اول هر سه H1، H2 و H3 است."
        )

        print(
            f"[SELECTED MODEL FOR H4] {selected_model}"
        )

    else:

        print(
            "[RESULT] هیچ مدلی در هر سه H1، H2 و H3 همزمان رتبه اول نشده است."
        )

        print(
            f"[SELECTED MODEL FOR H4] {selected_model}"
        )

        print(
            "انتخاب بر اساس تعداد رتبه‌های اول، مجموع رتبه‌ها و میانگین رتبه انجام شده است."
        )

    print("-" * 100)

    return ranking, selected_model, unanimous


# ==================================================================================================
# SAVE RESULTS
# ==================================================================================================

def save_outputs(
    ranking,
    selected_model,
    unanimous,
    h1_summary,
    h2_summary,
    h3_summary,
    h3_condition_wins,
):

    H4_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------------------------------------------
    # Final ranking
    # ----------------------------------------------------------------------------------------------

    ranking.to_csv(
        H4_RANKING,
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------------------------------

    summary_rows = []

    for _, row in ranking.iterrows():

        model = row["model"]

        h1_row = h1_summary[
            h1_summary["model"] == model
        ]

        h2_row = h2_summary[
            h2_summary["model"] == model
        ]

        h3_row = h3_summary[
            h3_summary["model"] == model
        ]

        result = {
            "model": model,

            "H1_mean_F1": (
                float(h1_row.iloc[0]["mean_f1"])
                if not h1_row.empty else np.nan
            ),

            "H1_rank": (
                int(h1_row.iloc[0]["H1_rank"])
                if not h1_row.empty else np.nan
            ),

            "H2_mean_degradation": (
                float(h2_row.iloc[0]["mean_degradation"])
                if not h2_row.empty else np.nan
            ),

            "H2_rank": (
                int(h2_row.iloc[0]["H2_rank"])
                if not h2_row.empty else np.nan
            ),

            "H3_mean_degradation": (
                float(h3_row.iloc[0]["mean_degradation"])
                if not h3_row.empty else np.nan
            ),

            "H3_rank": (
                int(h3_row.iloc[0]["H3_rank"])
                if not h3_row.empty else np.nan
            ),

            "H3_condition_wins": int(
                row["H3_condition_wins"]
            ),

            "H3_condition_mean_rank": float(
                row["H3_condition_mean_rank"]
            )
            if pd.notna(row["H3_condition_mean_rank"])
            else np.nan,

            "n_first_place": int(
                row["n_first_place"]
            ),

            "rank_sum": float(
                row["rank_sum"]
            ),

            "mean_rank": float(
                row["mean_rank"]
            ),

            "final_rank": int(
                row["final_rank"]
            ),

            "selected_for_H4": (
                model == selected_model
            ),
        }

        summary_rows.append(result)

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        H4_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------------------------------------------------------------
    # Selected model TXT
    # ----------------------------------------------------------------------------------------------

    with open(
        H4_SELECTED,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            selected_model
        )
        f.write("\n")

    # ----------------------------------------------------------------------------------------------
    # JSON details
    # ----------------------------------------------------------------------------------------------

    details = {
        "project": str(PROJECT),

        "input_files": {
            "H1": str(H1_FILE),
            "H2": str(H2_FILE),
            "H3": str(H3_FILE),
        },

        "output_files": {
            "summary": str(H4_SUMMARY),
            "ranking": str(H4_RANKING),
            "h3_condition_ranking": str(
                H4_H3_CONDITION_RANKING
            ),
            "h3_condition_wins": str(
                H4_H3_CONDITION_WINS
            ),
            "selected_model": str(H4_SELECTED),
        },

        "selection_rule": {
            "priority_1": "number_of_first_place_results_descending",
            "priority_2": "rank_sum_ascending",
            "priority_3": "mean_rank_ascending",
            "priority_4": "H3_condition_wins_descending",
            "priority_5": "H3_condition_mean_rank_ascending",
        },

        "metric_direction": {
            "H1": "higher_mean_F1_is_better",
            "H2": "lower_mean_degradation_is_better",
            "H3": "lower_mean_degradation_is_better",
        },

        "selected_model": selected_model,

        "unanimous_H1_H2_H3_winner": unanimous,

        "models": MODELS,
    }

    with open(
        H4_DETAILS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ----------------------------------------------------------------------------------------------
    # Print output files
    # ----------------------------------------------------------------------------------------------

    print()
    print_header("OUTPUT FILES")

    print(f"[1] {H4_SUMMARY}")
    print(f"[2] {H4_RANKING}")
    print(f"[3] {H4_H3_CONDITION_RANKING}")
    print(f"[4] {H4_H3_CONDITION_WINS}")
    print(f"[5] {H4_SELECTED}")
    print(f"[6] {H4_DETAILS}")


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print_header(
        "H4 MODEL SELECTION FROM H1 + H2 + H3"
    )

    print(f"Project directory:")
    print(PROJECT)

    # ----------------------------------------------------------------------------------------------
    # Validate input files
    # ----------------------------------------------------------------------------------------------

    print_header("INPUT FILE VALIDATION")

    validate_file(
        H1_FILE,
        "H1 per-image metrics"
    )

    validate_file(
        H2_FILE,
        "H2 per-image degradation"
    )

    validate_file(
        H3_FILE,
        "H3 all-conditions per-image domain shift"
    )

    # ----------------------------------------------------------------------------------------------
    # H1
    # ----------------------------------------------------------------------------------------------

    h1_summary = process_h1()

    # ----------------------------------------------------------------------------------------------
    # H2
    # ----------------------------------------------------------------------------------------------

    h2_summary = process_h2()

    # ----------------------------------------------------------------------------------------------
    # H3
    # ----------------------------------------------------------------------------------------------

    h3_df, h3_summary = process_h3()

    # ----------------------------------------------------------------------------------------------
    # H3 condition balanced
    # ----------------------------------------------------------------------------------------------

    h3_condition_summary, h3_condition_wins = (
        process_h3_conditions(h3_df)
    )

    # ----------------------------------------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------------------------------------

    ranking, selected_model, unanimous = combine_results(
        h1_summary,
        h2_summary,
        h3_summary,
        h3_condition_wins,
    )

    # ----------------------------------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------------------------------

    save_outputs(
        ranking,
        selected_model,
        unanimous,
        h1_summary,
        h2_summary,
        h3_summary,
        h3_condition_wins,
    )

    # ----------------------------------------------------------------------------------------------
    # Final
    # ----------------------------------------------------------------------------------------------

    print()
    print_header("H4 MODEL SELECTION COMPLETED")

    print(
        f"Selected model for H4: {selected_model}"
    )

    if unanimous:

        print(
            "این مدل در هر سه H1، H2 و H3 رتبه اول را کسب کرده است."
        )

    else:

        print(
            "این مدل بر اساس معیار تجمیعی رتبه‌بندی برای H4 انتخاب شده است."
        )

    print()
    print("Done.")


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 100)
        print("ERROR")
        print("=" * 100)
        print(str(exc))
        print("=" * 100)

        sys.exit(1)

