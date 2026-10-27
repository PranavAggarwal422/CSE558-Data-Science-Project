from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIRMATION_FILE = PROJECT_ROOT / "data" / "processed" / "confirmation.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

ALPHA = 0.05

# Smallest effects we decided could matter in practice.
DURATION_SME = 1.0       # minutes
DISTANCE_SME = 0.25      # miles


def welch_test(
    group_a: pd.Series,
    group_b: pd.Series,
    test_name: str,
    estimand: str,
    practical_threshold: float,
) -> dict:
    """Compare two independent groups without assuming equal variances.

    The reported estimate is mean(group_a) - mean(group_b).
    """
    a = group_a.dropna().astype(float)
    b = group_b.dropna().astype(float)

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    estimate = mean_a - mean_b
    standard_error = np.sqrt(var_a / n_a + var_b / n_b)

    # Welch-Satterthwaite degrees of freedom.
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (
        (var_a / n_a) ** 2 / (n_a - 1)
        + (var_b / n_b) ** 2 / (n_b - 1)
    )
    degrees_of_freedom = numerator / denominator

    test_statistic = estimate / standard_error
    p_value = 2 * stats.t.sf(abs(test_statistic), df=degrees_of_freedom)
    critical_value = stats.t.ppf(1 - ALPHA / 2, df=degrees_of_freedom)
    ci_low = estimate - critical_value * standard_error
    ci_high = estimate + critical_value * standard_error

    return {
        "test": test_name,
        "method": "Welch two-sample t-test",
        "estimand": estimand,
        "n_group_a": n_a,
        "n_group_b": n_b,
        "estimate": estimate,
        "standard_error": standard_error,
        "test_statistic_t": test_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value_two_sided": p_value,
        "ci95_lower": ci_low,
        "ci95_upper": ci_high,
        "alpha": ALPHA,
        "smallest_practically_meaningful_effect": practical_threshold,
        "statistically_significant": p_value < ALPHA,
        "ci_entirely_beyond_practical_threshold": (
            ci_low > practical_threshold or ci_high < -practical_threshold
        ),
    }


def paired_t_test(
    differences: pd.Series,
    test_name: str,
    estimand: str,
    practical_threshold: float,
) -> dict:
    """Test whether the mean within-date difference is zero."""
    d = differences.dropna().astype(float)
    n = len(d)
    estimate = d.mean()
    standard_deviation = d.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(n)
    degrees_of_freedom = n - 1

    test_statistic = estimate / standard_error
    p_value = 2 * stats.t.sf(abs(test_statistic), df=degrees_of_freedom)
    critical_value = stats.t.ppf(1 - ALPHA / 2, df=degrees_of_freedom)
    ci_low = estimate - critical_value * standard_error
    ci_high = estimate + critical_value * standard_error

    return {
        "test": test_name,
        "method": "Paired t-test",
        "estimand": estimand,
        "n_dates": n,
        "estimate": estimate,
        "standard_error": standard_error,
        "test_statistic_t": test_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value_two_sided": p_value,
        "ci95_lower": ci_low,
        "ci95_upper": ci_high,
        "alpha": ALPHA,
        "smallest_practically_meaningful_effect": practical_threshold,
        "statistically_significant": p_value < ALPHA,
        "ci_entirely_beyond_practical_threshold": (
            ci_low > practical_threshold or ci_high < -practical_threshold
        ),
    }

def payment_type_chi_square_test(con, confirmation_path):
    """Test association between identified payment types and day category."""

    path = str(confirmation_path.resolve()).replace("'", "''")

    # Payment type 5 is documented as Unknown.
    # Exclude it only from this test, not from the prepared dataset.
    query = f"""
        SELECT
            payment_type,
            CASE
                WHEN DAYOFWEEK(CAST(tpep_pickup_datetime AS DATE)) IN (0, 6)
                THEN 'Weekend'
                ELSE 'Weekday'
            END AS day_category,
            COUNT(*) AS trips
        FROM read_parquet('{path}')
        WHERE payment_type IN (0, 1, 2, 3, 4)
        GROUP BY payment_type, day_category
        ORDER BY payment_type, day_category
    """

    counts = con.execute(query).fetchdf()

    observed_df = (
        counts.pivot(
            index="payment_type",
            columns="day_category",
            values="trips"
        )
        .reindex(index=[0, 1, 2, 3, 4], columns=["Weekday", "Weekend"])
        .fillna(0)
        .astype(int)
    )

    observed = observed_df.to_numpy()

    chi2, p_value, degrees_of_freedom, expected = (
        stats.chi2_contingency(observed, correction=False)
    )

    total_trips = int(observed.sum())
    min_expected = float(expected.min())

    # Display the within-day percentages to aid interpretation.
    percentages = observed_df.div(observed_df.sum(axis=0), axis=1) * 100

    print("\n" + "=" * 76)
    print("T4: PAYMENT TYPE AND WEEKDAY/WEEKEND")
    print("=" * 76)
    print("\nObserved counts:")
    print(observed_df.to_string())

    print("\nPercentages within each day category:")
    print(percentages.round(3).to_string())

    print(f"\nTrips included: {total_trips:,}")
    print("Payment type 5 (Unknown) excluded from this test: 1 record")
    print(f"Chi-square statistic: {chi2:.6f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"Minimum expected count: {min_expected:.2f}")
    print(f"P-value: {p_value:.6g}")
    print(f"Decision at alpha = {ALPHA}: "
          f"{'Reject H0' if p_value < ALPHA else 'Fail to reject H0'}")

    return {
        "test": "T4: Payment type vs day category",
        "method": "Chi-square test of independence",
        "estimand": "Association between payment type and day category",
        "n_total": total_trips,
        "test_statistic_chi_square": chi2,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": ALPHA,
        "statistically_significant": p_value < ALPHA,
        "minimum_expected_count": min_expected,
        "unknown_type_5_excluded": 1,
    }

def main():
    if not CONFIRMATION_FILE.exists():
        raise FileNotFoundError(
            f"Confirmation dataset not found: {CONFIRMATION_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = str(CONFIRMATION_FILE.resolve()).replace("'", "''")

    # First calculate one set of daily summaries from the confirmation data.
    # Sunday = 0 and Saturday = 6 in DuckDB.
    con = duckdb.connect()
    try:
        daily = con.execute(
            f"""
            WITH per_day AS (
                SELECT
                    CAST(tpep_pickup_datetime AS DATE) AS trip_date,
                    AVG(duration_minutes) AS mean_duration,
                    AVG(trip_distance) AS mean_distance,
                    AVG(duration_minutes) FILTER (
                        WHERE HOUR(tpep_pickup_datetime) BETWEEN 14 AND 17
                    ) AS afternoon_mean_duration,
                    AVG(duration_minutes) FILTER (
                        WHERE HOUR(tpep_pickup_datetime) BETWEEN 0 AND 3
                    ) AS overnight_mean_duration
                FROM read_parquet('{path}')
                GROUP BY CAST(tpep_pickup_datetime AS DATE)
            )
            SELECT
                trip_date,
                DAYOFWEEK(trip_date) AS weekday_number,
                mean_duration,
                mean_distance,
                afternoon_mean_duration,
                overnight_mean_duration
            FROM per_day
            ORDER BY trip_date
            """
        ).fetchdf()
    finally:
        con.close()

    daily["is_weekend"] = daily["weekday_number"].isin([0, 6])

    # Test 1: independent groups of dates (weekend dates vs weekday dates).
    t1 = welch_test(
        group_a=daily.loc[daily["is_weekend"], "mean_duration"],
        group_b=daily.loc[~daily["is_weekend"], "mean_duration"],
        test_name="T1: Weekend vs weekday duration",
        estimand="Weekend minus weekday daily mean duration (minutes)",
        practical_threshold=DURATION_SME,
    )

    # Test 2: paired comparison, because both time periods are measured
    # on the same calendar date. Keep dates with data in both time periods.
    paired = daily.dropna(
        subset=["afternoon_mean_duration", "overnight_mean_duration"]
    ).copy()
    differences = (
        paired["afternoon_mean_duration"] - paired["overnight_mean_duration"]
    )
    t2 = paired_t_test(
        differences=differences,
        test_name="T2: Afternoon vs overnight duration",
        estimand="Afternoon minus overnight daily mean duration (minutes)",
        practical_threshold=DURATION_SME,
    )

    # Test 3: independent groups of dates (weekend dates vs weekday dates).
    t3 = welch_test(
        group_a=daily.loc[daily["is_weekend"], "mean_distance"],
        group_b=daily.loc[~daily["is_weekend"], "mean_distance"],
        test_name="T3: Weekend vs weekday distance",
        estimand="Weekend minus weekday daily mean distance (miles)",
        practical_threshold=DISTANCE_SME,
    )
    
    # Test 4: association between payment type and day category.
    t4_con = duckdb.connect()
    try:
        t4 = payment_type_chi_square_test(
            con=t4_con,
            confirmation_path=CONFIRMATION_FILE,
        )
    finally:
        t4_con.close()

    results = pd.DataFrame([t1, t2, t3, t4])
    output_file = OUTPUT_DIR / "hypothesis_test_results.csv"
    results.to_csv(output_file, index=False)

    print("=" * 76)
    print("HYPOTHESIS TESTS (SIMPLE T-TEST METHODS)")
    print("=" * 76)
    print(f"Confirmation file: {CONFIRMATION_FILE}")
    print(f"Calendar dates in daily summary: {len(daily)}")
    print(f"Dates used in paired comparison: {len(paired)}")
    print(f"Significance level: {ALPHA}")
    print("\nResults:")
    display_columns = [
        "test",
        "method",
        "estimate",
        "test_statistic_t",
        "test_statistic_chi_square",
        "degrees_of_freedom",
        "p_value_two_sided",
        "p_value",
        "ci95_lower",
        "ci95_upper",
        "statistically_significant",
        "minimum_expected_count",
        "unknown_type_5_excluded",
        "ci_entirely_beyond_practical_threshold",
    ]
    print(results[display_columns].to_string(index=False))
    print(f"\nSaved results to: {output_file}")
    print(
        "\nNote: These tests compare daily summaries, not individual trips. "
        "The conclusions describe differences in daily averages."
    )


if __name__ == "__main__":
    main()
