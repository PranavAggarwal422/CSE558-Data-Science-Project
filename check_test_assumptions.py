from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path.cwd()
DATA = ROOT / "data" / "processed" / "confirmation.parquet"
OUT_DIR = ROOT / "results" / "figures" / "assumption_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"Could not find {DATA}. Run this script from your project root: "
        r"C:\Semester 5\DSc Project"
    )

df = pd.read_parquet(DATA)

required = {"tpep_pickup_datetime", "duration_minutes", "trip_distance"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in confirmation.parquet: {sorted(missing)}")

df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"], errors="coerce"
)
df = df.dropna(subset=["tpep_pickup_datetime", "duration_minutes", "trip_distance"]).copy()
df["date"] = df["tpep_pickup_datetime"].dt.date
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["is_weekend"] = df["tpep_pickup_datetime"].dt.dayofweek >= 5

# Re-create the daily-level quantities used in the three tests.
daily = df.groupby("date").agg(
    daily_mean_duration=("duration_minutes", "mean"),
    daily_mean_distance=("trip_distance", "mean"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["is_weekend"] = daily["date"].dt.dayofweek >= 5

afternoon = (
    df[df["hour"].between(14, 17)]
    .groupby("date")["duration_minutes"].mean()
    .rename("afternoon_mean")
)
overnight = (
    df[df["hour"].between(0, 3)]
    .groupby("date")["duration_minutes"].mean()
    .rename("overnight_mean")
)
paired = pd.concat([afternoon, overnight], axis=1).dropna()
paired["difference"] = paired["afternoon_mean"] - paired["overnight_mean"]

samples = {
    "T1 weekday daily mean duration": daily.loc[~daily["is_weekend"], "daily_mean_duration"],
    "T1 weekend daily mean duration": daily.loc[daily["is_weekend"], "daily_mean_duration"],
    "T2 paired daily differences (afternoon - overnight)": paired["difference"],
    "T3 weekday daily mean distance": daily.loc[~daily["is_weekend"], "daily_mean_distance"],
    "T3 weekend daily mean distance": daily.loc[daily["is_weekend"], "daily_mean_distance"],
}

def describe(name, series):
    x = pd.Series(series).dropna().astype(float)
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
    print(f"\n{name}")
    print(f"  n={len(x)}; mean={x.mean():.4f}; SD={x.std(ddof=1):.4f}")
    print(f"  min={x.min():.4f}; Q1={q1:.4f}; median={x.median():.4f}; Q3={q3:.4f}; max={x.max():.4f}")
    print(f"  skewness={stats.skew(x, bias=False):.4f}; IQR-rule outliers={outliers}")
    if len(x) >= 3:
        # Shapiro-Wilk is limited to 5,000 observations; these samples are small.
        sw = stats.shapiro(x)
        print(f"  Shapiro-Wilk p-value={sw.pvalue:.6g} (diagnostic only; not a pass/fail rule)")

print("CONFIRMATION-DATA DIAGNOSTICS")
print(f"Rows loaded after removing missing required values: {len(df):,}")
print(f"Calendar dates in daily summary: {len(daily)}")
print(f"Dates available for paired comparison: {len(paired)}")
print("Weekend definition: Saturday and Sunday")
print("Afternoon hours: 14, 15, 16, 17; overnight hours: 0, 1, 2, 3")

for name, series in samples.items():
    describe(name, series)

# Save one histogram per test-relevant distribution.
for name, series in samples.items():
    x = pd.Series(series).dropna().astype(float)
    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "minus")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x, bins="auto", edgecolor="black")
    ax.set_title(name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of dates")
    fig.tight_layout()
    path = OUT_DIR / f"{safe_name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved histogram: {path.relative_to(ROOT)}")

# Save daily quantities and paired differences to make the checks reproducible.
daily.to_csv(ROOT / "results" / "tables" / "diagnostic_daily_summaries.csv", index=False)
paired.reset_index().to_csv(
    ROOT / "results" / "tables" / "diagnostic_paired_differences.csv", index=False
)
print("\nSaved diagnostic CSVs in results/tables/")
print("Finished. These checks describe the data; they do not automatically prove or disprove test assumptions.")
