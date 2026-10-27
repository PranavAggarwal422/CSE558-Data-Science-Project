from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLORATION_FILE = (
    PROJECT_ROOT / "data" / "processed" / "exploration.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPLORATION_FILE.exists():
        raise FileNotFoundError(EXPLORATION_FILE)

    con = duckdb.connect()

    path = str(EXPLORATION_FILE.resolve()).replace("'", "''")
    source = f"read_parquet('{path}')"

    # One row per calendar date.
    # These summaries will help us choose the inference design.
    daily = con.execute(
        f"""
        SELECT
            CAST(tpep_pickup_datetime AS DATE) AS trip_date,
            DAYOFWEEK(tpep_pickup_datetime) AS weekday_number,

            AVG(duration_minutes) AS mean_duration,
            MEDIAN(duration_minutes) AS median_duration,
            AVG(trip_distance) AS mean_distance,

            AVG(duration_minutes) FILTER (
                WHERE HOUR(tpep_pickup_datetime) BETWEEN 14 AND 17
            ) AS afternoon_mean_duration,

            AVG(duration_minutes) FILTER (
                WHERE HOUR(tpep_pickup_datetime) BETWEEN 0 AND 3
            ) AS overnight_mean_duration

        FROM {source}
        GROUP BY trip_date, weekday_number
        ORDER BY trip_date
        """
    ).fetchdf()

    daily.to_csv(
        OUTPUT_DIR / "daily_inference_summary.csv",
        index=False,
    )

    # DAYOFWEEK: Sunday = 0, Saturday = 6.
    weekend = daily["weekday_number"].isin([0, 6])
    weekday = ~weekend

    print("=" * 75)
    print("INFERENCE PLANNING — EXPLORATION DATA ONLY")
    print("=" * 75)

    print("\nNumber of calendar dates:")
    print(f"All dates: {len(daily)}")
    print(f"Weekdays: {weekday.sum()}")
    print(f"Weekend dates: {weekend.sum()}")

    print("\nDaily mean trip duration (minutes):")
    print(f"Weekdays: {daily.loc[weekday, 'mean_duration'].mean():.4f}")
    print(f"Weekend:  {daily.loc[weekend, 'mean_duration'].mean():.4f}")

    print("\nDaily median trip duration (minutes):")
    print(f"Weekdays: {daily.loc[weekday, 'median_duration'].mean():.4f}")
    print(f"Weekend:  {daily.loc[weekend, 'median_duration'].mean():.4f}")

    print("\nDaily mean trip distance (miles):")
    print(f"Weekdays: {daily.loc[weekday, 'mean_distance'].mean():.4f}")
    print(f"Weekend:  {daily.loc[weekend, 'mean_distance'].mean():.4f}")

    valid_periods = daily.dropna(
        subset=[
            "afternoon_mean_duration",
            "overnight_mean_duration",
        ]
    )

    daily_difference = (
        valid_periods["afternoon_mean_duration"]
        - valid_periods["overnight_mean_duration"]
    )

    print("\nAfternoon minus overnight mean duration:")
    print(f"Dates with both periods: {len(valid_periods)}")
    print(f"Mean daily difference: {daily_difference.mean():.4f} minutes")
    print(f"Median daily difference: {daily_difference.median():.4f} minutes")

    print("\nSaved:")
    print(OUTPUT_DIR / "daily_inference_summary.csv")

    con.close()


if __name__ == "__main__":
    main()
