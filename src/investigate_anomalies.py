from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

PREPARED_FILE = PROCESSED_DIR / "trips_prepared.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREPARED_FILE.exists():
        raise FileNotFoundError(PREPARED_FILE)

    con = duckdb.connect()

    path = str(PREPARED_FILE.resolve()).replace("'", "''")
    source = f"read_parquet('{path}')"

    print("=" * 75)
    print("TIMESTAMP AND EXTREME-DURATION INVESTIGATION")
    print("=" * 75)

    # 1. Check whether pickup timestamps agree with source months.
    monthly = con.execute(
        f"""
        SELECT
            source_month,
            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE strftime(tpep_pickup_datetime, '%Y-%m')
                      <> source_month
            ) AS pickup_month_mismatch,

            COUNT(*) FILTER (
                WHERE strftime(tpep_dropoff_datetime, '%Y-%m')
                      <> source_month
            ) AS dropoff_month_mismatch,

            MIN(tpep_pickup_datetime) AS earliest_pickup,
            MAX(tpep_pickup_datetime) AS latest_pickup,

            COUNT(*) FILTER (
                WHERE duration_minutes > 120
            ) AS duration_over_2_hours,

            COUNT(*) FILTER (
                WHERE duration_minutes > 360
            ) AS duration_over_6_hours,

            COUNT(*) FILTER (
                WHERE duration_minutes > 1440
            ) AS duration_over_24_hours

        FROM {source}
        GROUP BY source_month
        ORDER BY source_month
        """
    ).fetchdf()

    print("\nMonthly timestamp and duration checks:")
    print(monthly.to_string(index=False))

    monthly.to_csv(
        OUTPUT_DIR / "timestamp_anomaly_summary.csv",
        index=False,
    )

    # 2. Count records by pickup year and month.
    by_period = con.execute(
        f"""
        SELECT
            strftime(tpep_pickup_datetime, '%Y-%m') AS pickup_period,
            COUNT(*) AS records,
            MIN(tpep_pickup_datetime) AS earliest_pickup,
            MAX(tpep_pickup_datetime) AS latest_pickup
        FROM {source}
        GROUP BY pickup_period
        ORDER BY pickup_period
        """
    ).fetchdf()

    print("\nRecords grouped by actual pickup month:")
    print(by_period.to_string(index=False))

    by_period.to_csv(
        OUTPUT_DIR / "pickup_period_summary.csv",
        index=False,
    )

    # 3. Inspect the most extreme duration records.
    extreme = con.execute(
        f"""
        SELECT
            source_month,
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            duration_minutes,
            trip_distance,
            fare_amount,
            total_amount,
            VendorID,
            PULocationID,
            DOLocationID
        FROM {source}
        WHERE duration_minutes > 360
           OR strftime(tpep_pickup_datetime, '%Y-%m')
              <> source_month
        ORDER BY duration_minutes DESC
        LIMIT 50
        """
    ).fetchdf()

    print("\nExtreme-duration or pickup-month-mismatch records:")
    print(extreme.to_string(index=False))

    extreme.to_csv(
        OUTPUT_DIR / "extreme_duration_records.csv",
        index=False,
    )

    con.close()

    print("\nInvestigation files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
