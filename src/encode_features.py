from pathlib import Path
import csv
import json
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, vstack
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "trips_prepared.parquet"
OUTPUT_DIR = ROOT / "outputs" / "encoded"

CHUNK_SIZE = 100_000

# Nominal attributes: their numeric-looking values are category labels.
CATEGORICAL_COLUMNS = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "source_month",
]

# Numerical attributes retained in their original units.
NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
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
    "duration_minutes",
]

MISSING_TOKEN = "__MISSING__"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    con = duckdb.connect()
    input_path = str(INPUT_FILE.resolve()).replace("\\", "/").replace("'", "''")
    relation = f"read_parquet('{input_path}')"

    try:
        row_count = con.execute(
            f"SELECT COUNT(*) FROM {relation}"
        ).fetchone()[0]

        # Fit the categorical vocabulary from distinct observed values.
        # Missing values receive an explicit category.
        categories = []

        for column in CATEGORICAL_COLUMNS:
            quoted = '"' + column.replace('"', '""') + '"'

            values = con.execute(
                f"""
                SELECT DISTINCT CAST({quoted} AS VARCHAR)
                FROM {relation}
                WHERE {quoted} IS NOT NULL
                ORDER BY 1
                """
            ).fetchall()

            observed = [row[0] for row in values]

            if MISSING_TOKEN in observed:
                raise ValueError(
                    f"{MISSING_TOKEN!r} already occurs as a real value "
                    f"in {column!r}; choose another missing token."
                )

            if con.execute(
                f"SELECT COUNT(*) FROM {relation} WHERE {quoted} IS NULL"
            ).fetchone()[0] > 0:
                observed.append(MISSING_TOKEN)

            categories.append(observed)

        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="error",
            sparse_output=True,
            dtype=np.float32,
        )

        # Fit using one representative row per category. This creates the
        # vocabulary without loading the full dataset into memory.
        fit_rows = []
        for column, column_categories in zip(
            CATEGORICAL_COLUMNS, categories
        ):
            pass

        # Fit on a small synthetic frame containing all vocabulary values.
        # Each column's values are independent; the Cartesian product is
        # not needed.
        max_categories = max(len(values) for values in categories)
        fit_data = {}

        for column, column_categories in zip(
            CATEGORICAL_COLUMNS, categories
        ):
            padded = column_categories + [
                column_categories[0]
            ] * (max_categories - len(column_categories))
            fit_data[column] = padded

        fit_frame = pd.DataFrame(fit_data, dtype="string")
        encoder.fit(fit_frame)

        feature_names = list(encoder.get_feature_names_out())
        categorical_feature_count = len(feature_names)
        feature_names.extend(NUMERIC_COLUMNS)

        # Transform chunks and write a sparse matrix for each chunk.
        # Separate files avoid building a huge matrix in memory.
        chunk_files = []
        missing_counts = {column: 0 for column in NUMERIC_COLUMNS}
        numeric_rows = 0
        encoded_rows = 0

        select_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        select_sql = ", ".join(
            '"' + column.replace('"', '""') + '"'
            for column in select_columns
        )

        offset = 0
        chunk_number = 0

        while offset < row_count:
            query = f"""
                SELECT {select_sql}
                FROM {relation}
                LIMIT {CHUNK_SIZE} OFFSET {offset}
            """

            frame = con.execute(query).fetchdf()

            if frame.empty:
                break

            chunk_number += 1

            categorical = frame[CATEGORICAL_COLUMNS].copy()

            for column in CATEGORICAL_COLUMNS:
                categorical[column] = (
                    categorical[column]
                    .astype("string")
                    .fillna(MISSING_TOKEN)
                )

            encoded_categorical = encoder.transform(categorical)

            numeric = frame[NUMERIC_COLUMNS].to_numpy(
                dtype=np.float32,
                na_value=np.nan,
            )

            for index, column in enumerate(NUMERIC_COLUMNS):
                missing_counts[column] += int(
                    np.isnan(numeric[:, index]).sum()
                )

            # Preserve numerical missingness as NaN; do not impute.
            from scipy.sparse import csr_matrix, hstack

            numeric_sparse = csr_matrix(numeric, dtype=np.float32)
            combined = hstack(
                [encoded_categorical, numeric_sparse],
                format="csr",
                dtype=np.float32,
            )

            chunk_path = OUTPUT_DIR / f"chunk_{chunk_number:04d}.npz"
            save_npz(chunk_path, combined, compressed=True)
            chunk_files.append(chunk_path)

            encoded_rows += combined.shape[0]
            numeric_rows += len(frame)
            offset += len(frame)

            print(
                f"Processed chunk {chunk_number}: "
                f"{len(frame):,} rows; "
                f"total {encoded_rows:,}/{row_count:,}"
            )

        if encoded_rows != row_count:
            raise RuntimeError(
                f"Row mismatch: encoded {encoded_rows:,}, "
                f"expected {row_count:,}"
            )

        # Save feature-name mapping and encoding metadata.
        mapping_path = OUTPUT_DIR / "feature_names.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "categorical_columns": CATEGORICAL_COLUMNS,
                    "numeric_columns": NUMERIC_COLUMNS,
                    "categorical_feature_count": categorical_feature_count,
                    "total_feature_count": len(feature_names),
                    "feature_names": feature_names,
                    "missing_category_token": MISSING_TOKEN,
                    "numeric_missing_value_policy": (
                        "Preserve missing numerical values as NaN; "
                        "no imputation."
                    ),
                    "chunk_size": CHUNK_SIZE,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        audit_path = OUTPUT_DIR / "encoding_audit.csv"
        with audit_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["input_rows", row_count])
            writer.writerow(["encoded_rows", encoded_rows])
            writer.writerow(["input_attributes", 22])
            writer.writerow(
                ["categorical_input_attributes", len(CATEGORICAL_COLUMNS)]
            )
            writer.writerow(
                ["categorical_encoded_features", categorical_feature_count]
            )
            writer.writerow(
                ["numeric_features", len(NUMERIC_COLUMNS)]
            )
            writer.writerow(["total_encoded_features", len(feature_names)])
            writer.writerow(["chunk_count", len(chunk_files)])
            writer.writerow(["chunk_size", CHUNK_SIZE])
            writer.writerow(["input_rows_times_attributes", row_count * 22])
            writer.writerow(
                [
                    "encoded_rows_times_features",
                    encoded_rows * len(feature_names),
                ]
            )
            writer.writerow(
                [
                    "numeric_missing_values_preserved",
                    sum(missing_counts.values()),
                ]
            )
            writer.writerow(
                [
                    "output_bytes",
                    sum(path.stat().st_size for path in chunk_files),
                ]
            )
            writer.writerow(["elapsed_seconds", round(time.time() - started, 2)])

        print("\nEncoding complete.")
        print(f"Rows: {encoded_rows:,}")
        print(f"Categorical features: {categorical_feature_count:,}")
        print(f"Numerical features: {len(NUMERIC_COLUMNS):,}")
        print(f"Total features: {len(feature_names):,}")
        print(f"Chunks: {len(chunk_files):,}")
        print(f"Feature mapping: {mapping_path}")
        print(f"Audit: {audit_path}")
        print(
            "Compressed chunk output size: "
            f"{sum(path.stat().st_size for path in chunk_files) / (1024**2):,.2f} MiB"
        )

        print("\nNumeric missing values:")
        for column, count in missing_counts.items():
            if count:
                print(f"  {column}: {count:,}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
