from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPARED_FILE = PROCESSED_DIR / "trips_prepared.parquet"


def main() -> None:
    if not PREPARED_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {PREPARED_FILE}"
        )

    con = duckdb.connect()

    file_path = str(PREPARED_FILE.resolve()).replace("'", "''")
    relation = f"read_parquet('{file_path}')"

    print("=" * 75)
    print("PREPARED DATASET INSPECTION")
    print("=" * 75)

    # Record count and schema
    print("\nRecord count:")
    print(con.execute(
        f"SELECT COUNT(*) FROM {relation}"
    ).fetchone()[0])

    print("\nSchema:")
    print(con.execute(
        f"DESCRIBE SELECT * FROM {relation}"
    ).fetchdf().to_string(index=False))

    # Records by month
    print("\nRecords by source month:")
    print(con.execute(
        f"""
        SELECT
            source_month,
            COUNT(*) AS records,
            ROUND(AVG(duration_minutes), 3) AS mean_duration_minutes,
            ROUND(MEDIAN(duration_minutes), 3) AS median_duration_minutes
        FROM {relation}
        GROUP BY source_month
        ORDER BY source_month
        """
    ).fetchdf().to_string(index=False))

    # Duration distribution
    print("\nDuration summary:")
    print(con.execute(
        f"""
        SELECT
            MIN(duration_minutes) AS minimum,
            QUANTILE_CONT(duration_minutes, 0.01) AS p01,
            QUANTILE_CONT(duration_minutes, 0.25) AS q1,
            MEDIAN(duration_minutes) AS median,
            AVG(duration_minutes) AS mean,
            QUANTILE_CONT(duration_minutes, 0.75) AS q3,
            QUANTILE_CONT(duration_minutes, 0.99) AS p99,
            MAX(duration_minutes) AS maximum
        FROM {relation}
        """
    ).fetchdf().to_string(index=False))

    # Check for any unexpected violations of our initial rules
    print("\nValidation checks:")
    print(con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (
                WHERE duration_minutes <= 0
            ) AS nonpositive_duration,
            COUNT(*) FILTER (
                WHERE trip_distance <= 0
            ) AS nonpositive_distance,
            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime IS NULL
                   OR tpep_dropoff_datetime IS NULL
            ) AS missing_timestamps
        FROM {relation}
        """
    ).fetchdf().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
