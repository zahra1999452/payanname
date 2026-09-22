from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher"
    r"\\marine_debris_detection\hypothesis_tests\results2"
)

OUT = ROOT / "plots1"
OUT.mkdir(parents=True, exist_ok=True)

HYPOTHESES = ["H1", "H2", "H3"]

# ============================================================
# HELPERS
# ============================================================

def normalize_model_name(x):
    s = str(x).strip().lower()

    if "yolo" in s:
        return "YOLOv8n"

    if "rtdetr" in s or "rt-detr" in s:
        return "RT-DETR-L"

    if "rtmdet" in s:
        return "RTMDet-Tiny"

    return str(x)


def find_map_columns(df):
    """
    Find all columns related to mAP.
    """
    cols = []

    for c in df.columns:
        cs = str(c).strip().lower().replace(" ", "")

        if "map@0.50:0.95" in cs:
            cols.append(c)

        elif "map50-95" in cs:
            cols.append(c)

        elif "map@0.50" in cs:
            cols.append(c)

        elif "map50" in cs:
            cols.append(c)

        elif "map" in cs:
            cols.append(c)

    # preserve order, remove duplicates
    return list(dict.fromkeys(cols))


def identify_model_column(df):
    candidates = [
        "model",
        "Model",
        "MODEL",
        "model_name",
        "model_name",
        "detector",
        "network"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # fallback: search any column containing model
    for c in df.columns:
        if "model" in str(c).lower():
            return c

    return None


def get_map_type(column):
    s = str(column).lower().replace(" ", "")

    if "0.50:0.95" in s or "50-95" in s:
        return "mAP@0.50:0.95"

    if "0.50" in s or "map50" in s:
        return "mAP@0.50"

    return str(column)


def load_all_csvs(folder):
    files = list(folder.rglob("*.csv"))

    # exclude generated output folder
    files = [
        f for f in files
        if "MAP_PLOTS" not in f.parts
    ]

    return files


# ============================================================
# EXTRACT MAP
# ============================================================

def extract_maps(hypothesis):
    folder = ROOT / hypothesis

    print("\n" + "=" * 90)
    print(f"{hypothesis} — SEARCHING ALL CSV FILES FOR mAP")
    print("=" * 90)

    if not folder.exists():
        print("[ERROR] Folder does not exist:")
        print(folder)
        return pd.DataFrame()

    csv_files = load_all_csvs(folder)

    print(f"CSV files found: {len(csv_files)}")

    rows = []

    for csv_file in csv_files:

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[SKIP] {csv_file.name} -> {e}")
            continue

        map_cols = find_map_columns(df)

        if not map_cols:
            continue

        print(f"\n[MAP FOUND] {csv_file.relative_to(folder)}")
        print("Columns:", map_cols)

        model_col = identify_model_column(df)

        if model_col is None:
            print("[WARNING] Model column not found")
            print("Columns:", list(df.columns))
            continue

        print("Model column:", model_col)

        for map_col in map_cols:

            temp = df[[model_col, map_col]].copy()

            temp.columns = ["model", "value"]

            temp["value"] = pd.to_numeric(
                temp["value"],
                errors="coerce"
            )

            temp = temp.dropna(subset=["value"])

            if temp.empty:
                continue

            temp["model"] = temp["model"].apply(
                normalize_model_name
            )

            temp["metric"] = get_map_type(map_col)
            temp["source_file"] = str(
                csv_file.relative_to(folder)
            )
            temp["hypothesis"] = hypothesis

            rows.append(temp)

    if not rows:
        print(f"\n[WARNING] No usable mAP data found in {hypothesis}")
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    # --------------------------------------------------------
    # Remove suspicious duplicate rows coming from repeated
    # extraction of the same file/metric/model
    # --------------------------------------------------------

    result = result.drop_duplicates(
        subset=[
            "hypothesis",
            "model",
            "metric",
            "source_file"
        ]
    )

    return result


# ============================================================
# LOAD H1 / H2 / H3
# ============================================================

all_results = []

for h in HYPOTHESES:

    result = extract_maps(h)

    if not result.empty:

        out_csv = OUT / f"{h}_MAP_extracted.csv"

        result.to_csv(
            out_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nSaved: {out_csv}")

        all_results.append(result)


if not all_results:
    print("\n" + "=" * 90)
    print("NO mAP DATA FOUND")
    print("=" * 90)
    raise SystemExit


all_maps = pd.concat(
    all_results,
    ignore_index=True
)

all_maps.to_csv(
    OUT / "ALL_H1_H2_H3_MAP.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n")
print("=" * 90)
print("ALL EXTRACTED mAP DATA")
print("=" * 90)

print(
    all_maps[
        [
            "hypothesis",
            "model",
            "metric",
            "value",
            "source_file"
        ]
    ].to_string(index=False)
)


# ============================================================
# FUNCTION: PLOT ONE HYPOTHESIS
# ============================================================

def plot_hypothesis(hypothesis, metric):

    df = all_maps[
        (all_maps["hypothesis"] == hypothesis) &
        (all_maps["metric"] == metric)
    ].copy()

    if df.empty:
        print(
            f"[WARNING] No {metric} data for {hypothesis}"
        )
        return

    # --------------------------------------------------------
    # Aggregate if multiple rows exist for same model
    # --------------------------------------------------------

    summary = (
        df.groupby("model", as_index=False)["value"]
        .mean()
    )

    # Fixed model order
    model_order = [
        "YOLOv8n",
        "RT-DETR-L",
        "RTMDet-Tiny"
    ]

    summary["order"] = summary["model"].map(
        {m: i for i, m in enumerate(model_order)}
    )

    summary = (
        summary
        .sort_values("order")
        .drop(columns="order")
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        summary["model"],
        summary["value"]
    )

    plt.ylabel(metric)
    plt.xlabel("Model")

    plt.title(
        f"{hypothesis} — {metric}"
    )

    plt.ylim(
        0,
        max(
            1.0,
            float(summary["value"].max()) * 1.15
        )
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # Value labels
    for bar, value in zip(
        bars,
        summary["value"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    filename = (
        f"{hypothesis}_"
        f"{metric.replace('@', '').replace(':', '_')}.png"
    )

    path = OUT / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[PLOT] {path}")


# ============================================================
# CREATE H1 / H2 / H3 PLOTS
# ============================================================

metrics = [
    "mAP@0.50",
    "mAP@0.50:0.95"
]

for h in HYPOTHESES:

    for metric in metrics:

        plot_hypothesis(
            h,
            metric
        )


# ============================================================
# COMBINED PLOT
# ============================================================

def plot_combined(metric):

    df = all_maps[
        all_maps["metric"] == metric
    ].copy()

    if df.empty:
        print(
            f"[WARNING] No combined data for {metric}"
        )
        return

    pivot = (
        df.groupby(
            ["hypothesis", "model"],
            as_index=False
        )["value"]
        .mean()
        .pivot(
            index="hypothesis",
            columns="model",
            values="value"
        )
    )

    model_order = [
        "YOLOv8n",
        "RT-DETR-L",
        "RTMDet-Tiny"
    ]

    for m in model_order:
        if m not in pivot.columns:
            pivot[m] = np.nan

    pivot = pivot[model_order]

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_ylabel(metric)
    ax.set_xlabel("Experiment")
    ax.set_title(
        f"H1 / H2 / H3 — {metric}"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    filename = (
        f"COMBINED_H1_H2_H3_"
        f"{metric.replace('@', '').replace(':', '_')}.png"
    )

    path = OUT / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[COMBINED PLOT] {path}")


for metric in metrics:
    plot_combined(metric)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 90)
print("MAP PLOT GENERATION COMPLETED")
print("=" * 90)

print("\nOutput folder:")
print(OUT)

print("\nGenerated files should include:")

for f in sorted(OUT.glob("*")):
    print(" ", f.name)

print("\nDONE.")