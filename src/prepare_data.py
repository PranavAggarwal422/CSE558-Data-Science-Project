from pathlib import Path
import csv

import duckdb


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

EXPECTED_MONTHS = 6

# Keep only columns shared by all six monthly files.
COMMON_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]

# SEED = 2024422


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def sql_string(value: str) -> str:
    """Escape a string for use as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def execute_count(connection, query: str) -> int:
    """Execute a SQL query returning one integer."""
    return int(connection.execute(query).fetchone()[0])


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("yellow_tripdata_2026-*.parquet"))

    if len(files) != EXPECTED_MONTHS:
        raise RuntimeError(
            f"Expected {EXPECTED_MONTHS} monthly files, found {len(files)}. "
            "Check data/raw before proceeding."
        )

    print("Input files:")
    for path in files:
        print(f"  {path.name}: {path.stat().st_size / (1024 ** 2):.2f} MB")

    # Use a local DuckDB database file so processing is easier to inspect
    # and rerun than a large in-memory pandas workflow.
    database_path = PROCESSED_DIR / "preparation_work.duckdb"
    connection = duckdb.connect(str(database_path))

    # Create a harmonized view. The June-only request_source field is
    # intentionally excluded; the original files remain untouched.
    monthly_queries = []

    for path in files:
        month = path.stem[-7:]  # Example: 2026-01
        column_sql = ", ".join(f'"{column}"' for column in COMMON_COLUMNS)

        monthly_queries.append(
            f"""
            SELECT
                {sql_string(month)} AS source_month,
                {column_sql}
            FROM read_parquet({sql_string(str(path.resolve()))})
            """
        )

    union_sql = "\nUNION ALL\n".join(monthly_queries)

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW harmonized_trips AS
        {union_sql}
        """
    )

    raw_count = execute_count(
        connection, "SELECT COUNT(*) FROM harmonized_trips"
    )

    # Basic row counts before applying the analytical filters.
    timestamps_valid = execute_count(
        connection,
        """
        SELECT COUNT(*)
        FROM harmonized_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
        """,
    )

    positive_duration = execute_count(
        connection,
        """
        SELECT COUNT(*)
        FROM harmonized_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
          AND tpep_dropoff_datetime > tpep_pickup_datetime
        """,
    )

    positive_distance = execute_count(
        connection,
        """
        SELECT COUNT(*)
        FROM harmonized_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
          AND tpep_dropoff_datetime > tpep_pickup_datetime
          AND trip_distance > 0
        """,
    )

    pickup_month_match = execute_count(
        connection,
        """
        SELECT COUNT(*)
        FROM harmonized_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
          AND tpep_dropoff_datetime > tpep_pickup_datetime
          AND trip_distance > 0
          AND strftime(tpep_pickup_datetime, '%Y-%m') = source_month
        """,
    )

    # Build the primary analytical dataset.
    connection.execute(
        """
        CREATE OR REPLACE TABLE prepared_trips AS
        SELECT *
        FROM (
            SELECT
                *,
                date_diff(
                    'microsecond',
                    tpep_pickup_datetime,
                    tpep_dropoff_datetime
                ) / 60000000.0 AS duration_minutes
            FROM harmonized_trips
        )
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
          AND tpep_dropoff_datetime > tpep_pickup_datetime
          AND trip_distance > 0
          AND strftime(tpep_pickup_datetime, '%Y-%m') = source_month
          AND duration_minutes <= 360
        """
    )

    prepared_count = execute_count(
        connection, "SELECT COUNT(*) FROM prepared_trips"
    )

    # Save the complete prepared dataset.
    prepared_path = PROCESSED_DIR / "trips_prepared.parquet"

    connection.execute(
        f"""
        COPY prepared_trips
        TO {sql_string(str(prepared_path.resolve()))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Reserve a deterministic 20% confirmation set using a stable hash
    # of selected trip attributes. The same inputs and software version
    # produce the same partition.
    #
    # The hash is used for reproducible partitioning, not as a data
    # deduplication method. Records with identical selected attributes
    # will be assigned to the same partition.
    split_key = """
        hash(
            concat_ws(
                '|',
                CAST(tpep_pickup_datetime AS VARCHAR),
                CAST(tpep_dropoff_datetime AS VARCHAR),
                COALESCE(CAST(VendorID AS VARCHAR), '<NULL>'),
                COALESCE(CAST(PULocationID AS VARCHAR), '<NULL>'),
                COALESCE(CAST(DOLocationID AS VARCHAR), '<NULL>'),
                COALESCE(CAST(trip_distance AS VARCHAR), '<NULL>'),
                COALESCE(CAST(fare_amount AS VARCHAR), '<NULL>'),
                COALESCE(CAST(source_month AS VARCHAR), '<NULL>')
            )
        )
    """

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM prepared_trips
            WHERE ({split_key}) % 5 <> 0
        )
        TO {sql_string(str((PROCESSED_DIR / "exploration.parquet").resolve()))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM prepared_trips
            WHERE ({split_key}) % 5 = 0
        )
        TO {sql_string(str((PROCESSED_DIR / "confirmation.parquet").resolve()))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    exploration_count = execute_count(
        connection,
        f"SELECT COUNT(*) FROM prepared_trips WHERE ({split_key}) % 5 <> 0",
    )

    confirmation_count = execute_count(
        connection,
        f"SELECT COUNT(*) FROM prepared_trips WHERE ({split_key}) % 5 = 0",
    )

    # Record the number of rows at each preparation stage.
    audit_rows = [
        ("Raw records across all six files", raw_count),
        ("Records with both timestamps present", timestamps_valid),
        (
            "Records with positive duration",
            positive_duration,
        ),
        (
            "Records with positive duration and positive distance",
            positive_distance,
        ),
        (
            "Records also matching their source pickup month",
            pickup_month_match,
        ),
        (
            "Final prepared records after six-hour duration cutoff",
            prepared_count,
        ),
        ("Records in exploration partition", exploration_count),
        ("Records in confirmation partition", confirmation_count),
    ]

    audit_path = REPORTS_DIR / "preparation_audit.csv"

    with audit_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "record_count"])
        writer.writerows(audit_rows)

    # Capture missingness in the prepared dataset. Missingness is reported,
    # not automatically imputed or used as a reason to delete a row.
    missingness_queries = []

    for column in COMMON_COLUMNS:
        missingness_queries.append(
            f"""
            SELECT
                {sql_string(column)} AS column_name,
                COUNT(*) FILTER (WHERE "{column}" IS NULL) AS missing_count,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE "{column}" IS NULL)
                    / NULLIF(COUNT(*), 0),
                    4
                ) AS missing_percent
            FROM prepared_trips
            """
        )

    missingness_sql = "\nUNION ALL\n".join(missingness_queries)

    missingness = connection.execute(
        missingness_sql + " ORDER BY column_name"
    ).fetchdf()

    missingness.to_csv(
        REPORTS_DIR / "prepared_missingness.csv",
        index=False,
    )

    print("\nPreparation results")
    print("-" * 60)
    for stage, count in audit_rows:
        print(f"{stage}: {count:,}")

    print("\nPartition check:")
    print(
        f"Exploration + confirmation = "
        f"{exploration_count + confirmation_count:,}"
    )
    print(f"Prepared records = {prepared_count:,}")

    if exploration_count + confirmation_count != prepared_count:
        raise RuntimeError("Partition counts do not match prepared records.")

    print("\nFiles created:")
    print(f"  {prepared_path}")
    print(f"  {PROCESSED_DIR / 'exploration.parquet'}")
    print(f"  {PROCESSED_DIR / 'confirmation.parquet'}")
    print(f"  {audit_path}")
    print(f"  {REPORTS_DIR / 'prepared_missingness.csv'}")
    print(f"  {database_path}")

    connection.close()


if __name__ == "__main__":
    main()
