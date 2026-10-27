# NYC Yellow Taxi Trip Analysis

**Course:** CSE558 — Data Science  
**Semester:** Monsoon 2026  
**Team:** DataForge 1  
**Dataset:** NYC TLC Yellow Taxi Trip Records, January–June 2026

## 1. Overview

This project analyzes New York City Yellow Taxi trip records from January through June 2026. It combines data preparation, exploratory analysis, data-quality checks, feature encoding, and statistical hypothesis testing.

The final prepared dataset contains **21,846,625 trips and 22 attributes**. The project investigates differences in trip duration and distance and examines the association between payment type and weekday/weekend status.

## 2. Research Questions

The project investigates four questions:

1. **Weekend vs. weekday duration:** Does mean daily trip duration differ between weekend dates and weekday dates?
2. **Afternoon vs. overnight duration:** On the same date, does mean trip duration differ between trips picked up from 14:00–17:59 and trips picked up from 00:00–03:59?
3. **Weekend vs. weekday distance:** Does mean daily trip distance differ between weekend dates and weekday dates?
4. **Payment type and day type:** Is payment-type distribution associated with whether a trip occurs on a weekday or weekend?

The first three questions use date-level summaries from the confirmation partition. The fourth uses trip-level counts in a payment-type-by-weekday/weekend contingency table.

## 3. Dataset Download

**The raw and processed datasets are not included in this repository.** Download the source data separately before running the preparation pipeline.

### Source

Use the official NYC Taxi and Limousine Commission (TLC) Trip Record Data page:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Download the **Yellow Taxi Trip Records** for all six months of 2026:

- January 2026
- February 2026
- March 2026
- April 2026
- May 2026
- June 2026

Choose the Parquet files, not the CSV versions.

### Required directory structure

After downloading the files, place them in `data/raw/` using the following filenames:

```text
data/
└── raw/
    ├── yellow_tripdata_2026-01.parquet
    ├── yellow_tripdata_2026-02.parquet
    ├── yellow_tripdata_2026-03.parquet
    ├── yellow_tripdata_2026-04.parquet
    ├── yellow_tripdata_2026-05.parquet
    └── yellow_tripdata_2026-06.parquet
```

The preparation script expects exactly six files matching `yellow_tripdata_2026-*.parquet`. If the files are missing or the expected filenames do not match, preparation will stop with an error.

**Do not upload these datasets to GitHub.** The `.gitignore` excludes the raw and processed data directories.

## 4. Environment Setup

The commands below assume Windows PowerShell and Python are installed.

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The required Python packages and their pinned versions are listed in `requirements.txt`.

Run project commands from the repository root so that paths resolve correctly.

## 5. Reproducing the Project

### Step 1: Prepare the data

After downloading and placing all six Parquet files in `data/raw/`, run:

```powershell
python src/prepare_data.py
```

The preparation pipeline:

- Harmonizes the monthly schemas by retaining the 20 shared source attributes.
- Adds `source_month` and `duration_minutes`.
- Removes records that fail the documented duration, distance, source-month, and six-hour duration-cutoff rules.
- Creates a deterministic exploration/confirmation split.
- Generates preparation and missingness audit reports.

The principal outputs are written locally:

```text
data/processed/
├── trips_prepared.parquet
├── exploration.parquet
├── confirmation.parquet
└── preparation_work.duckdb

data/reports/
├── preparation_audit.csv
└── prepared_missingness.csv
```

These are generated files and are not committed to GitHub.

### Step 2: Run data-quality checks

```powershell
python src/quality_checks.py
python src/check_missingness_overlap.py
```

These scripts support validation of the prepared dataset and investigation of missing-value patterns.

### Step 3: Generate the data dictionary

```powershell
python src/build_data_dictionary.py
```

The generated data dictionary describes the prepared attributes, their types, missingness, observed values, and data-quality treatments.

### Step 4: Run exploratory analysis

```powershell
python src/exploratory_analysis.py
```

The exploratory analysis produces summaries and figures used to examine trip-duration, trip-distance, hourly, and monthly patterns.

Figures retained in the repository are under:

```text
results/figures/
```

### Step 5: Run hypothesis tests

```powershell
python src/hypothesis_tests.py
```

This script runs the project's four primary statistical tests. The results and supporting summaries are documented in the report and in the relevant CSV files under `results/tables/`.

### Step 6: Generate encoded features (optional)

```powershell
python src/encode_features.py
```

The encoding pipeline generates sparse feature matrices in chunks. It is separate from the main statistical analysis and is not required merely to inspect the report or its existing results.

The generated `.npz` chunks are intentionally excluded from GitHub because they occupy substantial storage space. The encoding audit and feature-name metadata are retained.

**Note:** The encoded representation preserves numeric missing values. It is not a fully imputed, model-ready dataset.

## 6. Data Preparation and Quality

The six source files contain **22,836,530 rows** in total. The June-only `request_source` field was excluded to harmonize the monthly schemas.

The prepared dataset contains **21,846,625 rows × 22 attributes**, retaining approximately **95.67%** of the original records.

The preparation process applies the following rules:

- Retain records with valid pickup and drop-off timestamps.
- Require positive trip duration and positive trip distance.
- Require the pickup month to match the source-file month.
- Apply the documented six-hour duration cutoff.
- Retain unusual monetary values where the available evidence does not justify blanket deletion.
- Preserve missing optional values rather than automatically imputing them.

Five attributes—`passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, and `Airport_fee`—are missing on exactly the same 5,327,931 rows (24.3879%). No rows have only a subset of these five fields missing.

A full-row MD5 hash screen found zero duplicate hashes. This is a practical screening check, not a mathematical guarantee against hash collisions.

## 7. Exploration and Confirmation Split

The prepared dataset is partitioned deterministically using a hash-based rule:

- **Exploration partition:** 17,479,462 rows
- **Confirmation partition:** 4,367,163 rows

The split uses selected trip attributes and a modulo-based rule. It does not depend on a random seed. Reproducing the partition requires the same input records, key construction, hashing implementation, and rule.

Exploratory summaries are used to describe the data, while the primary hypothesis tests use the confirmation partition.

## 8. Feature Encoding

The encoding workflow converts seven categorical attributes into 550 one-hot features and retains 13 numeric features, resulting in **563 encoded features**.

The process preserves all 21,846,625 prepared rows across 219 sparse chunks. Timestamp fields are not included in this encoded matrix, and numeric missing values are not imputed.

The encoded chunks are generated locally rather than stored in GitHub.

## 9. Statistical Findings

The primary hypothesis tests use a significance level of **0.05**.

| Test | Method | Main result |
|---|---|---|
| T1: Weekend vs. weekday duration | Welch two-sample t-test on date-level means | Weekend-minus-weekday estimate: −1.971 minutes; 95% CI [−2.350, −1.593]; p < 0.001. Reject the null hypothesis. |
| T2: Afternoon vs. overnight duration | Paired t-test by date | Afternoon-minus-overnight estimate: +5.085 minutes; 95% CI [4.752, 5.418]; p < 0.001. Reject the null hypothesis. |
| T3: Weekend vs. weekday distance | Welch two-sample t-test on date-level means | Weekend-minus-weekday estimate: −0.449 miles; 95% CI [−1.772, 0.873]; p = 0.502. Fail to reject the null hypothesis. |
| T4: Payment type vs. weekday/weekend | Pearson chi-square test of independence | χ² = 55,583.686; df = 4; p < 0.001. Reject independence. |

### Distance sensitivity analysis

The distance analysis is sensitive to extreme recorded distances. In a separate sensitivity analysis restricted to trips of at most 50 miles, the weekend-minus-weekday estimate becomes **+0.101 miles** (95% CI [0.005, 0.196], p = 0.0388).

This sensitivity result is reported separately. It does not replace the primary T3 result or change the prepared dataset.

These findings describe associations in the observed data and do not establish causal effects.

## 10. Repository Structure

The repository contains source code, documentation, figures, and compact result tables. Large datasets and generated intermediate files are excluded.

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── CSE558_Project_Report.tex
├── CSE558_Project_Report.pdf
├── src/
├── results/
│   ├── figures/
│   │   └── assumption_checks/
│   └── tables/
└── outputs/
    └── encoded/
        ├── encoding_audit.csv
        └── feature_names.json
```

### Files intentionally excluded

- Raw monthly taxi Parquet files
- Prepared datasets and exploration/confirmation partitions
- Intermediate DuckDB database
- Generated encoded `.npz` chunks
- Python virtual environments and caches
- Temporary files and backup scripts

## 11. Team Contributions

- **Pranav Aggarwal (2024422):** Data preparation, feature encoding, exploratory analysis, Tests 1–2, and shared report writing.
- **Nirvan (2024390):** Data-quality audit, sensitivity analysis, Tests 3–4, and shared report writing.

## 12. Limitations and Reproducibility Notes

- The source data covers January–June 2026; findings should not automatically be generalized to other periods.
- The six-hour duration cutoff is a documented analytical decision and does not prove that every excluded trip is invalid.
- Extreme recorded values can influence distance summaries.
- Missing optional fields are preserved rather than imputed.
- The encoded representation is not a fully imputed model-ready matrix.


## 13. Use of LLM Tools
Large language models (LLMs) were used as programming aids to help implement selected Python functionalities, understand coding approaches, and debug implementation issues. Their assistance was limited to code-level support in response to functionalities and requirements specified by the project authors.

The project's data-processing decisions, exploratory data analysis (EDA) strategy, statistical hypotheses and test selection, validation procedures, interpretation of results, and overall analytical conclusions were determined by the project authors. LLM-generated code suggestions were reviewed, adapted where necessary, and cross-verified against the intended functionality, data, and analytical requirements before being incorporated into the project. LLMs were not treated as authoritative sources for analytical decisions or conclusions.

