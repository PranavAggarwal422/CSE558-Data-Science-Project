from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

EXPLORATION_FILE = PROCESSED_DIR / "exploration.parquet"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPLORATION_FILE.exists():
        raise FileNotFoundError(
            f"Exploration dataset not found: {EXPLORATION_FILE}"
        )

    con = duckdb.connect()

    path = str(EXPLORATION_FILE.resolve()).replace("'", "''")
    relation = f"read_parquet('{path}')"

    print("=" * 75)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 75)

    exploration_count = con.execute(
        f"SELECT COUNT(*) FROM {relation}"
    ).fetchone()[0]

    print(f"Exploration records: {exploration_count:,}")

    # -----------------------------------------------------
    # 1. Duration diagnostics
    # -----------------------------------------------------

    duration_summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS records,
            AVG(duration_minutes) AS mean_minutes,
            MEDIAN(duration_minutes) AS median_minutes,
            QUANTILE_CONT(duration_minutes, 0.90) AS p90,
            QUANTILE_CONT(duration_minutes, 0.95) AS p95,
            QUANTILE_CONT(duration_minutes, 0.99) AS p99,
            MAX(duration_minutes) AS maximum_minutes,
            COUNT(*) FILTER (
                WHERE duration_minutes > 120
            ) AS over_2_hours,
            COUNT(*) FILTER (
                WHERE duration_minutes > 360
            ) AS over_6_hours,
            COUNT(*) FILTER (
                WHERE duration_minutes > 1440
            ) AS over_24_hours
        FROM {relation}
        """
    ).fetchdf()

    duration_summary.to_csv(
        TABLES_DIR / "duration_diagnostics.csv",
        index=False,
    )

    print("\nDuration diagnostics:")
    print(duration_summary.to_string(index=False))

    # -----------------------------------------------------
    # 2. Duration histogram
    # -----------------------------------------------------

    # Use a bounded range for readability, while reporting
    # extreme observations separately in duration_diagnostics.csv.
    duration_data = con.execute(
        f"""
        SELECT duration_minutes
        FROM {relation}
        WHERE duration_minutes <= 120
        """
    ).fetchdf()["duration_minutes"]

    plt.figure(figsize=(9, 5))
    plt.hist(duration_data, bins=60)
    plt.xlabel("Trip duration (minutes)")
    plt.ylabel("Number of trips")
    plt.title("Distribution of Trip Duration (0–120 Minutes)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "duration_distribution.png",
        dpi=180,
    )
    plt.close()

    # -----------------------------------------------------
    # 3. Trip distance distribution
    # -----------------------------------------------------

    distance_data = con.execute(
        f"""
        SELECT trip_distance
        FROM {relation}
        WHERE trip_distance > 0
          AND trip_distance <= 50
        """
    ).fetchdf()["trip_distance"]

    plt.figure(figsize=(9, 5))
    plt.hist(distance_data, bins=60)
    plt.xlabel("Trip distance (miles)")
    plt.ylabel("Number of trips")
    plt.title("Distribution of Trip Distance (0–50 Miles)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "distance_distribution.png",
        dpi=180,
    )
    plt.close()

    # -----------------------------------------------------
    # 4. Duration by month
    # -----------------------------------------------------

    monthly = con.execute(
        f"""
        SELECT
            source_month,
            COUNT(*) AS records,
            AVG(duration_minutes) AS mean_duration,
            MEDIAN(duration_minutes) AS median_duration
        FROM {relation}
        GROUP BY source_month
        ORDER BY source_month
        """
    ).fetchdf()

    monthly.to_csv(
        TABLES_DIR / "monthly_duration_summary.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        monthly["source_month"],
        monthly["mean_duration"],
        marker="o",
        label="Mean",
    )
    plt.plot(
        monthly["source_month"],
        monthly["median_duration"],
        marker="o",
        label="Median",
    )
    plt.xlabel("Month")
    plt.ylabel("Trip duration (minutes)")
    plt.title("Trip Duration by Month")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "monthly_duration.png",
        dpi=180,
    )
    plt.close()

    # -----------------------------------------------------
    # 5. Duration by pickup hour
    # -----------------------------------------------------

    hourly = con.execute(
        f"""
        SELECT
            HOUR(tpep_pickup_datetime) AS pickup_hour,
            COUNT(*) AS records,
            AVG(duration_minutes) AS mean_duration,
            MEDIAN(duration_minutes) AS median_duration
        FROM {relation}
        GROUP BY pickup_hour
        ORDER BY pickup_hour
        """
    ).fetchdf()

    hourly.to_csv(
        TABLES_DIR / "hourly_duration_summary.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        hourly["pickup_hour"],
        hourly["mean_duration"],
        marker="o",
        label="Mean",
    )
    plt.plot(
        hourly["pickup_hour"],
        hourly["median_duration"],
        marker="o",
        label="Median",
    )
    plt.xlabel("Pickup hour (0–23)")
    plt.ylabel("Trip duration (minutes)")
    plt.title("Trip Duration by Pickup Hour")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "hourly_duration.png",
        dpi=180,
    )
    plt.close()

    print("\nEDA complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")

    con.close()


if __name__ == "__main__":
    main()
