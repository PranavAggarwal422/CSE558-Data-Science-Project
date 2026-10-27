from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_file(path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"QUALITY CHECKS: {path.name}")
    print("=" * 80)

    # Read metadata and schema without loading the complete file.
    parquet_file = pq.ParquetFile(path)
    print(f"Rows: {parquet_file.metadata.num_rows:,}")
    print(f"Columns: {parquet_file.metadata.num_columns}")
    print("Schema fields:", parquet_file.schema_arrow.names)

    # Load one month at a time.
    df = pd.read_parquet(path)

    # Calculate duration in minutes.
    df["duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    print("\nMissingness:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(
            pd.DataFrame(
                {
                    "missing_count": missing,
                    "missing_percent": (
                        missing / len(df) * 100
                    ).round(3),
                }
            ).to_string()
        )

    # Check whether selected missing values occur together.
    related = [
        c for c in [
            "passenger_count",
            "RatecodeID",
            "store_and_fwd_flag",
            "congestion_surcharge",
            "Airport_fee",
        ]
        if c in df.columns
    ]

    if related:
        print("\nMissingness pattern for selected columns:")
        patterns = df[related].isna().value_counts(dropna=False)
        print(patterns.head(10).to_string())

    print("\nTrip duration checks:")
    print(f"Negative durations: {(df['duration_minutes'] < 0).sum():,}")
    print(f"Zero durations: {(df['duration_minutes'] == 0).sum():,}")
    print(
        "Durations over 6 hours: "
        f"{(df['duration_minutes'] > 360).sum():,}"
    )
    print(
        "Durations over 24 hours: "
        f"{(df['duration_minutes'] > 1440).sum():,}"
    )

    print("\nDistance checks:")
    print(f"Zero or negative distance: {(df['trip_distance'] <= 0).sum():,}")
    print(f"Distance over 100 miles: {(df['trip_distance'] > 100).sum():,}")

    print("\nFare checks:")
    print(f"Negative fare: {(df['fare_amount'] < 0).sum():,}")
    print(f"Zero fare: {(df['fare_amount'] == 0).sum():,}")

    print("\nPassenger count checks:")
    if "passenger_count" in df.columns:
        print(
            "Non-missing passenger count <= 0: "
            f"{(df['passenger_count'].notna() & (df['passenger_count'] <= 0)).sum():,}"
        )
        print(
            "Non-missing passenger count > 6: "
            f"{(df['passenger_count'].notna() & (df['passenger_count'] > 6)).sum():,}"
        )

    if "request_source" in df.columns:
        print("\nJune request_source:")
        print(df["request_source"].value_counts(dropna=False).head(15).to_string())

    del df


def main() -> None:
    files = sorted(RAW_DIR.glob("yellow_tripdata_2026-*.parquet"))

    if not files:
        raise FileNotFoundError(f"No 2026 Parquet files found in {RAW_DIR}")

    print(f"Found {len(files)} files.")

    for path in files:
        inspect_file(path)


if __name__ == "__main__":
    main()
