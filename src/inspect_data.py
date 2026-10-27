from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_file(file_path: Path) -> None:
    """Print basic metadata and data-quality information for a Parquet file."""

    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    # Physical file information
    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    # Read Parquet metadata without loading the entire dataset into memory
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata

    print(f"Rows: {metadata.num_rows:,}")
    print(f"Columns: {metadata.num_columns}")
    print(f"Row groups: {metadata.num_row_groups}")

    print("\nSchema:")
    print(parquet_file.schema)

    # Load the file to calculate missingness and inspect values.
    # We'll assess memory requirements before combining all months.
    df = pd.read_parquet(file_path)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(3)

    missing_summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct,
    })

    print(missing_summary.to_string())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nDuplicate rows:")
    print(f"{df.duplicated().sum():,}")

    print("\nNumeric summary:")
    print(df.describe().to_string())

    del df


def main() -> None:
    files = sorted(RAW_DIR.glob("yellow_tripdata_2026-*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No monthly Parquet files found in {RAW_DIR}. "
            "Download the dataset files first."
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Files found: {len(files)}")

    for file_path in files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()
