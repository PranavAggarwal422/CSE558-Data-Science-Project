from pathlib import Path
import duckdb

root = Path(__file__).resolve().parent
data_file = root / "data" / "processed" / "trips_prepared.parquet"
output_file = root / "results" / "tables" / "duplicate_check.csv"

if not data_file.exists():
    raise FileNotFoundError(data_file)

con = duckdb.connect()

path = str(data_file.resolve()).replace("\\", "/").replace("'", "''")

query = f"""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT md5(to_json(t))) AS distinct_row_hashes
FROM read_parquet('{path}') AS t
"""

total_rows, distinct_hashes = con.execute(query).fetchone()

result = {
    "dataset": "trips_prepared.parquet",
    "total_rows": total_rows,
    "distinct_row_hashes": distinct_hashes,
    "duplicate_rows_by_hash_count": total_rows - distinct_hashes,
}

output_file.parent.mkdir(parents=True, exist_ok=True)

import csv
with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=result.keys())
    writer.writeheader()
    writer.writerow(result)

print(result)
print(f"Saved to: {output_file}")

con.close()
