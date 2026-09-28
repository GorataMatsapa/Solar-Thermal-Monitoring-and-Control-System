
"""
STMaCS SYSTEM 455
FEATURE-TARGET ANALYSIS

Purpose
-------
Analyze the relationship between engineered System 455 features
and future Buffer Top Temperature targets.

This script DOES NOT:
- modify the feature dataset
- delete observations
- fill missing values
- train a final ML model
- perform train/test splitting

It DOES:
- calculate Pearson correlations
- calculate correlations with future temperature change
- identify highly redundant features
- identify features with missing values
- calculate mutual information
- compare feature relationships across prediction horizons
- save analysis tables for later model development
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "system455_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

TARGETS = [
    "target_buffer_top_5min",
    "target_buffer_top_15min",
    "target_buffer_top_30min",
    "target_buffer_top_60min",
]

DELTA_TARGETS = [
    "target_delta_buffer_top_5min",
    "target_delta_buffer_top_15min",
    "target_delta_buffer_top_30min",
    "target_delta_buffer_top_60min",
]

# Features that must never be used as predictors.
EXCLUDED_COLUMNS = {
    "timestamp",
    "Date",
    "Time",
}

# Future targets must NEVER become predictors.
EXCLUDED_PREFIXES = (
    "target_",
)

# Correlation threshold for redundancy analysis.
REDUNDANCY_THRESHOLD = 0.95


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def is_target_column(column):
    return column.startswith(EXCLUDED_PREFIXES)


# =============================================================================
# LOAD DATA
# =============================================================================

section("STMaCS SYSTEM 455 FEATURE-TARGET ANALYSIS")

print("Loading feature-engineered dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFeature dataset not found:\n{INPUT_FILE}\n\n"
        "Run feature_engineering.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df):,}")


# =============================================================================
# TIMESTAMP
# =============================================================================

section("TIMESTAMP VALIDATION")

if "timestamp" not in df.columns:
    raise ValueError(
        "The feature dataset must contain a 'timestamp' column."
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError(
        "Invalid timestamps detected. Resolve this before ML analysis."
    )

print("[OK] All timestamps are valid.")


# =============================================================================
# IDENTIFY TARGETS
# =============================================================================

section("TARGET IDENTIFICATION")

for target in TARGETS:
    if target in df.columns:
        print(f"[FOUND] {target}")
    else:
        print(f"[MISSING] {target}")

for target in DELTA_TARGETS:
    if target in df.columns:
        print(f"[FOUND] {target}")
    else:
        print(f"[MISSING] {target}")


# =============================================================================
# IDENTIFY CANDIDATE FEATURES
# =============================================================================

section("DEFINING CANDIDATE FEATURES")

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

feature_columns = []

for column in numeric_columns:

    if column in EXCLUDED_COLUMNS:
        continue

    if is_target_column(column):
        continue

    feature_columns.append(column)

print(f"Numeric columns:          {len(numeric_columns)}")
print(f"Candidate predictor cols: {len(feature_columns)}")

print("\nCandidate predictors:")

for feature in feature_columns:
    print(f"  - {feature}")


# =============================================================================
# FEATURE COMPLETENESS
# =============================================================================

section("FEATURE COMPLETENESS")

completeness_records = []

for feature in feature_columns:

    missing = df[feature].isna().sum()
    total = len(df)

    completeness = 100 * (1 - missing / total)

    completeness_records.append(
        {
            "feature": feature,
            "missing": missing,
            "complete_percent": completeness,
        }
    )

completeness_df = pd.DataFrame(completeness_records)

completeness_df = completeness_df.sort_values(
    "missing",
    ascending=False
)

print(
    f"{'Feature':45}"
    f"{'Missing':>12}"
    f"{'Complete %':>15}"
)

print("-" * 75)

for _, row in completeness_df.iterrows():

    print(
        f"{row['feature']:45}"
        f"{int(row['missing']):>12,}"
        f"{row['complete_percent']:>14.2f}%"
    )


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

section("PEARSON CORRELATION WITH FUTURE BUFFER TOP TEMPERATURE")

correlation_tables = {}

for target in TARGETS:

    if target not in df.columns:
        continue

    print(f"\nTarget: {target}")

    records = []

    for feature in feature_columns:

        valid = df[[feature, target]].dropna()

        if len(valid) < 10:
            correlation = np.nan
        else:
            correlation = valid[feature].corr(
                valid[target]
            )

        records.append(
            {
                "feature": feature,
                "correlation": correlation,
                "absolute_correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                ),
                "valid_samples": len(valid),
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "absolute_correlation",
        ascending=False
    )

    correlation_tables[target] = result

    print(
        f"{'Feature':45}"
        f"{'Correlation':>15}"
    )

    print("-" * 65)

    for _, row in result.head(20).iterrows():

        print(
            f"{row['feature']:45}"
            f"{row['correlation']:>15.4f}"
        )


# =============================================================================
# CORRELATION WITH FUTURE TEMPERATURE CHANGE
# =============================================================================

section("CORRELATION WITH FUTURE BUFFER TOP TEMPERATURE CHANGE")

delta_correlation_tables = {}

for target in DELTA_TARGETS:

    if target not in df.columns:
        continue

    print(f"\nTarget: {target}")

    records = []

    for feature in feature_columns:

        valid = df[[feature, target]].dropna()

        if len(valid) < 10:
            correlation = np.nan
        else:
            correlation = valid[feature].corr(
                valid[target]
            )

        records.append(
            {
                "feature": feature,
                "correlation": correlation,
                "absolute_correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                ),
                "valid_samples": len(valid),
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "absolute_correlation",
        ascending=False
    )

    delta_correlation_tables[target] = result

    print(
        f"{'Feature':45}"
        f"{'Correlation':>15}"
    )

    print("-" * 65)

    for _, row in result.head(20).iterrows():

        print(
            f"{row['feature']:45}"
            f"{row['correlation']:>15.4f}"
        )


# =============================================================================
# CORRELATION MATRIX BETWEEN FEATURES
# =============================================================================

section("FEATURE REDUNDANCY ANALYSIS")

feature_matrix = df[feature_columns].copy()

feature_corr = feature_matrix.corr()

redundant_pairs = []

for i in range(len(feature_corr.columns)):

    feature_a = feature_corr.columns[i]

    for j in range(i + 1, len(feature_corr.columns)):

        feature_b = feature_corr.columns[j]

        correlation = feature_corr.iloc[i, j]

        if pd.notna(correlation):

            if abs(correlation) >= REDUNDANCY_THRESHOLD:

                redundant_pairs.append(
                    {
                        "feature_1": feature_a,
                        "feature_2": feature_b,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                    }
                )

redundancy_df = pd.DataFrame(redundant_pairs)

if len(redundancy_df) == 0:

    print(
        f"No feature pairs exceeded the "
        f"{REDUNDANCY_THRESHOLD:.2f} correlation threshold."
    )

else:

    redundancy_df = redundancy_df.sort_values(
        "absolute_correlation",
        ascending=False
    )

    print(
        f"Highly correlated feature pairs "
        f"(>|{REDUNDANCY_THRESHOLD:.2f}|): "
        f"{len(redundancy_df)}"
    )

    print()

    print(
        f"{'Feature 1':40}"
        f"{'Feature 2':40}"
        f"{'Correlation':>15}"
    )

    print("-" * 100)

    for _, row in redundancy_df.head(50).iterrows():

        print(
            f"{row['feature_1']:40}"
            f"{row['feature_2']:40}"
            f"{row['correlation']:>15.4f}"
        )


# =============================================================================
# MUTUAL INFORMATION
# =============================================================================

section("MUTUAL INFORMATION ANALYSIS")

print(
    "Mutual information is used to identify nonlinear relationships "
    "between predictors and future Buffer Top Temperature."
)

mi_tables = {}

# Use the 60-minute target as the main nonlinear screening target.
main_target = "target_buffer_top_60min"

if main_target in df.columns:

    print(f"\nMain target: {main_target}")

    mi_records = []

    for feature in feature_columns:

        subset = df[[feature, main_target]].dropna()

        if len(subset) < 100:

            mi_value = np.nan

        else:

            X = subset[[feature]].values
            y = subset[main_target].values

            try:

                mi = mutual_info_regression(
                    X,
                    y,
                    random_state=42
                )

                mi_value = float(mi[0])

            except Exception:

                mi_value = np.nan

        mi_records.append(
            {
                "feature": feature,
                "mutual_information": mi_value,
                "valid_samples": len(subset),
            }
        )

    mi_df = pd.DataFrame(mi_records)

    mi_df = mi_df.sort_values(
        "mutual_information",
        ascending=False
    )

    mi_tables[main_target] = mi_df

    print(
        f"{'Feature':45}"
        f"{'Mutual Information':>22}"
    )

    print("-" * 70)

    for _, row in mi_df.head(25).iterrows():

        value = row["mutual_information"]

        if pd.isna(value):
            value_text = "NaN"
        else:
            value_text = f"{value:.4f}"

        print(
            f"{row['feature']:45}"
            f"{value_text:>22}"
        )


# =============================================================================
# HORIZON COMPARISON
# =============================================================================

section("FEATURE RANKING ACROSS PREDICTION HORIZONS")

horizon_records = []

for target, table in correlation_tables.items():

    horizon = target.replace(
        "target_buffer_top_",
        ""
    )

    for rank, (_, row) in enumerate(
        table.iterrows(),
        start=1
    ):

        horizon_records.append(
            {
                "horizon": horizon,
                "feature": row["feature"],
                "correlation": row["correlation"],
                "absolute_correlation": row["absolute_correlation"],
                "rank": rank,
            }
        )

horizon_df = pd.DataFrame(horizon_records)

if not horizon_df.empty:

    pivot = horizon_df.pivot(
        index="feature",
        columns="horizon",
        values="correlation"
    )

    rank_pivot = horizon_df.pivot(
        index="feature",
        columns="horizon",
        values="rank"
    )

    print("\nTop features for each horizon:")

    for horizon in sorted(
        horizon_df["horizon"].unique()
    ):

        subset = horizon_df[
            horizon_df["horizon"] == horizon
        ].sort_values("rank")

        print(f"\n{horizon}")

        for _, row in subset.head(10).iterrows():

            print(
                f"  {int(row['rank']):2d}. "
                f"{row['feature']:<40} "
                f"{row['correlation']:+.4f}"
            )

else:

    pivot = pd.DataFrame()
    rank_pivot = pd.DataFrame()


# =============================================================================
# IMPORTANT FEATURE CATEGORIES
# =============================================================================

section("FEATURE CATEGORY SUMMARY")

categories = {
    "Current-state": [],
    "Lag": [],
    "Rate-of-change": [],
    "Thermal-difference": [],
    "Time": [],
    "Other": [],
}

for feature in feature_columns:

    if "_lag_" in feature:
        categories["Lag"].append(feature)

    elif "_delta_" in feature:
        categories["Rate-of-change"].append(feature)

    elif (
        "stratification" in feature
        or "minus_" in feature
        or "delta_t" in feature
    ):
        categories["Thermal-difference"].append(feature)

    elif (
        feature in [
            "hour",
            "minute",
            "day_of_week",
            "day_of_year",
        ]
        or feature.endswith("_sin")
        or feature.endswith("_cos")
    ):
        categories["Time"].append(feature)

    elif feature in [
        "buffer_top",
        "buffer_centre",
        "buffer_bottom",
        "collector_temp",
        "solar_radiation",
        "ambient_temp",
        "solar_flow",
        "solar_power",
        "solar_temp_hot",
        "solar_temp_cold",
        "hotwater_flow",
        "hotwater_power",
        "hotwater_temp_hot",
        "hotwater_temp_cold",
        "pwm",
        "solar_pump",
        "electric_heater",
    ]:
        categories["Current-state"].append(feature)

    else:
        categories["Other"].append(feature)


for category, features in categories.items():

    print(
        f"{category:<25}: {len(features):>3} features"
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

section("SAVING ANALYSIS RESULTS")

# Feature completeness
completeness_file = OUTPUT_DIR / "feature_completeness.csv"
completeness_df.to_csv(
    completeness_file,
    index=False
)

print(f"[SAVED] {completeness_file}")


# Feature correlation matrices
correlation_matrix_file = (
    OUTPUT_DIR / "feature_correlation_matrix.csv"
)

feature_corr.to_csv(
    correlation_matrix_file
)

print(f"[SAVED] {correlation_matrix_file}")


# Redundant feature pairs
redundancy_file = (
    OUTPUT_DIR / "highly_correlated_features.csv"
)

redundancy_df.to_csv(
    redundancy_file,
    index=False
)

print(f"[SAVED] {redundancy_file}")


# Horizon correlations
horizon_file = (
    OUTPUT_DIR / "feature_target_correlations.csv"
)

horizon_df.to_csv(
    horizon_file,
    index=False
)

print(f"[SAVED] {horizon_file}")


# Rank matrix
rank_file = (
    OUTPUT_DIR / "feature_target_ranks.csv"
)

rank_pivot.to_csv(
    rank_file
)

print(f"[SAVED] {rank_file}")


# Mutual information
if mi_tables:

    mi_file = (
        OUTPUT_DIR
        / "mutual_information_60min.csv"
    )

    mi_tables[main_target].to_csv(
        mi_file,
        index=False
    )

    print(f"[SAVED] {mi_file}")


# Individual horizon correlation files
for target, table in correlation_tables.items():

    horizon = target.replace(
        "target_buffer_top_",
        ""
    )

    output_file = (
        OUTPUT_DIR
        / f"correlation_{horizon}.csv"
    )

    table.to_csv(
        output_file,
        index=False
    )

    print(f"[SAVED] {output_file}")


# Individual delta-target files
for target, table in delta_correlation_tables.items():

    horizon = target.replace(
        "target_delta_buffer_top_",
        ""
    )

    output_file = (
        OUTPUT_DIR
        / f"delta_correlation_{horizon}.csv"
    )

    table.to_csv(
        output_file,
        index=False
    )

    print(f"[SAVED] {output_file}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("FEATURE-TARGET ANALYSIS COMPLETE")

print(f"Input dataset:")
print(f"  {INPUT_FILE}")

print("\nAnalysis output directory:")
print(f"  {OUTPUT_DIR}")

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):

    print(f"  - {file.name}")


print("\nImportant:")
print("  - No source data was modified.")
print("  - No rows were deleted.")
print("  - No missing values were filled.")
print("  - No weather data was added.")
print("  - No train/test split was performed.")
print("  - Future target columns were excluded from predictors.")

print("\nNext stage:")
print("  1. Review feature-target relationships.")
print("  2. Decide treatment of ambient temperature.")
print("  3. Define the final predictor set.")
print("  4. Perform chronological train/test splitting.")
print("  5. Train baseline ML models.")

print("\nREADY FOR FEATURE SELECTION AND BASELINE MODEL DEVELOPMENT.")
print("=" * 80)

