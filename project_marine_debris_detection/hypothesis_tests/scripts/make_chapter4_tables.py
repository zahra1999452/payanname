from pathlib import Path
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

STAT_DIR = PROJECT / "results" / "statistics"

OUTPUT_DIR = (
    PROJECT /
    "results" /
    "chapter4_tables"
)


def read_if_exists(path):

    if not path.exists():
        return None

    return pd.read_csv(path)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = [
        "H1_statistics.csv",
        "H2_statistics.csv",
        "H3_statistics.csv",
        "H4_statistics.csv",
    ]

    for filename in files:

        source = STAT_DIR / filename

        df = read_if_exists(source)

        if df is None:
            print(
                f"Not available yet: {source}"
            )
            continue

        output = (
            OUTPUT_DIR /
            filename.replace(
                ".csv",
                "_chapter4.csv",
            )
        )

        df.to_csv(
            output,
            index=False,
        )

        print(
            f"Created: {output}"
        )


if __name__ == "__main__":
    main()