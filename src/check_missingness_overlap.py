"""Check whether the five prepared-data fields with missing values overlap exactly."""
import duckdb

DATA = "data/processed/trips_prepared.parquet"
query = f"""
SELECT
  COUNT(*) FILTER (WHERE passenger_count IS NULL OR RatecodeID IS NULL
    OR store_and_fwd_flag IS NULL OR congestion_surcharge IS NULL
    OR Airport_fee IS NULL) AS rows_missing_any,
  COUNT(*) FILTER (WHERE passenger_count IS NULL AND RatecodeID IS NULL
    AND store_and_fwd_flag IS NULL AND congestion_surcharge IS NULL
    AND Airport_fee IS NULL) AS rows_missing_all_five,
  COUNT(*) FILTER (WHERE
    (passenger_count IS NULL)::INT + (RatecodeID IS NULL)::INT
    + (store_and_fwd_flag IS NULL)::INT + (congestion_surcharge IS NULL)::INT
    + (Airport_fee IS NULL)::INT BETWEEN 1 AND 4) AS rows_missing_some_but_not_all_five
FROM read_parquet('{DATA}')
"""
print(duckdb.sql(query).df().to_string(index=False))
