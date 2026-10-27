from pathlib import Path
import pandas as pd
from scipy import stats

# Load the confirmation dataset
data_path = Path("data/processed/confirmation.parquet")
df = pd.read_parquet(data_path)

# Extract pickup date and identify weekends
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"]
)
df["date"] = df["tpep_pickup_datetime"].dt.date
df["is_weekend"] = (
    df["tpep_pickup_datetime"].dt.dayofweek >= 5
)

# Calculate daily average trip distance
daily = (
    df.groupby(["date", "is_weekend"])["trip_distance"]
    .mean()
    .reset_index(name="daily_mean_distance")
)

weekdays = daily.loc[~daily["is_weekend"], "daily_mean_distance"]
weekends = daily.loc[daily["is_weekend"], "daily_mean_distance"]

print("=" * 65)
print("WEEKEND VS WEEKDAY TRIP DISTANCE INVESTIGATION")
print("=" * 65)

# Compare the means and medians
print("\n1. SUMMARY STATISTICS")
summary = daily.groupby("is_weekend")["daily_mean_distance"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
summary.index = ["Weekdays", "Weekends"]
print(summary.round(4).to_string())

# Display dates with the highest average trip distances
print("\n2. TEN DATES WITH THE HIGHEST DAILY MEAN DISTANCE")
print(
    daily.nlargest(10, "daily_mean_distance")[
        ["date", "is_weekend", "daily_mean_distance"]
    ].to_string(index=False)
)

# Display dates with the lowest average trip distances
print("\n3. TEN DATES WITH THE LOWEST DAILY MEAN DISTANCE")
print(
    daily.nsmallest(10, "daily_mean_distance")[
        ["date", "is_weekend", "daily_mean_distance"]
    ].to_string(index=False)
)

# Compare the original Welch test with a simple sensitivity check.
# The sensitivity check excludes the three highest-distance dates
# overall ONLY to assess their influence, not to clean the dataset.
original = stats.ttest_ind(
    weekends, weekdays, equal_var=False
)

extreme_dates = set(
    daily.nlargest(3, "daily_mean_distance")["date"]
)
sensitivity_daily = daily[~daily["date"].isin(extreme_dates)]

weekdays_sens = sensitivity_daily.loc[
    ~sensitivity_daily["is_weekend"], "daily_mean_distance"
]
weekends_sens = sensitivity_daily.loc[
    sensitivity_daily["is_weekend"], "daily_mean_distance"
]

sensitivity = stats.ttest_ind(
    weekends_sens, weekdays_sens, equal_var=False
)

print("\n4. ORIGINAL WELCH T-TEST")
print(f"Weekend minus weekday mean: {weekends.mean() - weekdays.mean():.4f} miles")
print(f"t-statistic: {original.statistic:.4f}")
print(f"p-value: {original.pvalue:.6g}")

print("\n5. SENSITIVITY CHECK")
print("For investigation only: the three highest-distance dates are excluded.")
print(f"Dates excluded: {sorted(map(str, extreme_dates))}")
print(
    "Weekend minus weekday mean: "
    f"{weekends_sens.mean() - weekdays_sens.mean():.4f} miles"
)
print(f"t-statistic: {sensitivity.statistic:.4f}")
print(f"p-value: {sensitivity.pvalue:.6g}")

print("\nNote: This sensitivity check does not justify deleting these dates.")
print("We will use it only to understand how much they influence the result.")