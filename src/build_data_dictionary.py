from pathlib import Path
import csv
import json

import duckdb


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "trips_prepared.parquet"
OUTPUT_FILE = ROOT / "results" / "tables" / "data_dictionary.csv"

DESCRIPTIONS = {
    "source_month": (
        "Month of the source file",
        "nominal",
        "YYYY-MM",
    ),
    "VendorID": (
        "Taxi technology/vendor identifier",
        "nominal",
        "category",
    ),
    "tpep_pickup_datetime": (
        "Trip pickup date and time",
        "temporal",
        "datetime",
    ),
    "tpep_dropoff_datetime": (
        "Trip drop-off date and time",
        "temporal",
        "datetime",
    ),
    "passenger_count": (
        "Reported passenger count",
        "discrete",
        "passengers",
    ),
    "trip_distance": (
        "Reported trip distance",
        "continuous",
        "miles",
    ),
    "RatecodeID": (
        "Rate code identifier",
        "nominal",
        "category",
    ),
    "store_and_fwd_flag": (
        "Whether the trip record was held before being sent",
        "nominal",
        "category",
    ),
    "PULocationID": (
        "Taxi Zone identifier for pickup",
        "nominal",
        "zone ID",
    ),
    "DOLocationID": (
        "Taxi Zone identifier for drop-off",
        "nominal",
        "zone ID",
    ),
    "payment_type": (
        "Payment method/type code",
        "nominal",
        "category",
    ),
    "fare_amount": ("Base trip fare", "continuous", "USD"),
    "extra": ("Additional fare charges", "continuous", "USD"),
    "mta_tax": ("MTA tax component", "continuous", "USD"),
    "tip_amount": ("Reported tip amount", "continuous", "USD"),
    "tolls_amount": ("Reported toll charges", "continuous", "USD"),
    "improvement_surcharge": (
        "Improvement surcharge",
        "continuous",
        "USD",
    ),
    "total_amount": (
        "Reported total trip amount",
        "continuous",
        "USD",
    ),
    "congestion_surcharge": (
        "Congestion surcharge",
        "continuous",
        "USD",
    ),
    "Airport_fee": ("Airport fee", "continuous", "USD"),
    "cbd_congestion_fee": (
        "Central business district congestion fee",
        "continuous",
        "USD",
    ),
    "duration_minutes": (
        "Elapsed time between pickup and drop-off",
        "continuous",
        "minutes",
    ),
}

TREATMENTS = {
    "tpep_pickup_datetime": (
        "Required; rows with missing pickup timestamps are excluded."
    ),
    "tpep_dropoff_datetime": (
        "Required; rows with missing or non-increasing drop-off "
        "timestamps are excluded."
    ),
    "trip_distance": (
        "Rows with nonpositive distance are excluded. Extreme positive "
        "values are retained and flagged for data-quality review."
    ),
    "source_month": (
        "Used to verify pickup month against the source file."
    ),
    "duration_minutes": (
        "Derived from pickup and drop-off timestamps; durations "
        "over 360 minutes are excluded."
    ),
    "passenger_count": (
        "Missing values retained and reported; no blanket imputation."
    ),
    "RatecodeID": (
        "Missing values retained and reported; no blanket imputation."
    ),
    "store_and_fwd_flag": (
        "Missing values retained and reported; no blanket imputation."
    ),
    "congestion_surcharge": (
        "Missing values retained and reported; no blanket imputation."
    ),
    "Airport_fee": (
        "Missing values retained and reported; no blanket imputation."
    ),
    "fare_amount": (
        "Negative and unusual values retained; no blanket deletion."
    ),
    "total_amount": (
        "Negative and unusual values retained; no blanket deletion."
    ),
}


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    con = duckdb.connect()

    try:
        path = str(DATA_FILE.resolve()).replace("\\", "/").replace("'", "''")
        relation = f"read_parquet('{path}')"

        schema = con.execute(
            f"DESCRIBE SELECT * FROM {relation}"
        ).fetchdf()

        total_rows = con.execute(
            f"SELECT COUNT(*) FROM {relation}"
        ).fetchone()[0]

        rows = []

        for _, item in schema.iterrows():
            name = item["column_name"]
            dtype = item["column_type"]

            description, stat_type, unit = DESCRIPTIONS.get(
                name,
                (
                    "Description requires review",
                    "requires review",
                    "requires review",
                ),
            )

            column = quote_identifier(name)

            missing_count = con.execute(
                f"SELECT COUNT(*) FROM {relation} WHERE {column} IS NULL"
            ).fetchone()[0]

            missing_percent = (
                100.0 * missing_count / total_rows if total_rows else 0.0
            )

            distinct_count = con.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM {relation}"
            ).fetchone()[0]

            observed_min = ""
            observed_max = ""
            observed_categories = ""

            if stat_type in ("nominal", "ordinal"):
                categories = con.execute(
                    f"""
                    SELECT CAST({column} AS VARCHAR) AS value,
                           COUNT(*) AS n
                    FROM {relation}
                    WHERE {column} IS NOT NULL
                    GROUP BY 1
                    ORDER BY n DESC, value
                    LIMIT 20
                    """
                ).fetchall()

                observed_categories = json.dumps(
                    {
                        "note": (
                            "Top 20 categories by frequency; "
                            "not a complete category list."
                        ),
                        "categories": [
                            {"value": value, "count": count}
                            for value, count in categories
                        ],
                    },
                    ensure_ascii=False,
                )

            elif stat_type == "temporal":
                minimum, maximum = con.execute(
                    f"""
                    SELECT MIN({column}), MAX({column})
                    FROM {relation}
                    """
                ).fetchone()

                observed_min = str(minimum) if minimum is not None else ""
                observed_max = str(maximum) if maximum is not None else ""

            else:
                minimum, maximum = con.execute(
                    f"""
                    SELECT MIN({column}), MAX({column})
                    FROM {relation}
                    """
                ).fetchone()

                observed_min = str(minimum) if minimum is not None else ""
                observed_max = str(maximum) if maximum is not None else ""

            data_quality_note = ""

            if name == "trip_distance":
                extreme_count = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {relation}
                    WHERE {column} > 100
                    """
                ).fetchone()[0]

                data_quality_note = (
                    f"{extreme_count:,} records exceed 100 miles; "
                    "extreme values require interpretation. "
                    "They are retained in the prepared data."
                )

            elif name in ("fare_amount", "total_amount"):
                unusual_count = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {relation}
                    WHERE {column} < 0
                    """
                ).fetchone()[0]

                data_quality_note = (
                    f"{unusual_count:,} records have negative values; "
                    "these were retained rather than blanket-deleted."
                )

            elif missing_count > 0:
                data_quality_note = (
                    "Contains missing values; see missing_count and "
                    "missing_percent. No blanket imputation was applied."
                )

            rows.append(
                {
                    "column_name": name,
                    "description": description,
                    "duckdb_type": dtype,
                    "statistical_type": stat_type,
                    "unit": unit,
                    "missing_count": missing_count,
                    "missing_percent": round(missing_percent, 4),
                    "distinct_count": distinct_count,
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "observed_categories": observed_categories,
                    "data_quality_note": data_quality_note,
                    "treatment": TREATMENTS.get(
                        name,
                        "Retained in the prepared dataset; "
                        "field-specific handling depends on the analysis.",
                    ),
                }
            )

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Prepared dataset rows: {total_rows:,}")
        print(f"Dictionary attributes: {len(rows)}")
        print(f"Saved: {OUTPUT_FILE}")

        print("\nColumns and detected types:")
        for row in rows:
            print(
                f"{row['column_name']}: {row['duckdb_type']} | "
                f"{row['statistical_type']} | "
                f"distinct={row['distinct_count']:,} | "
                f"missing={row['missing_count']:,}"
            )

        unclassified = [
            row["column_name"]
            for row in rows
            if row["statistical_type"] == "requires review"
        ]

        if unclassified:
            print("\nWARNING: Attributes requiring review:", unclassified)
        else:
            print("\nAll attributes have descriptions and statistical types.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
