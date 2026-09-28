
"""
STMaCS SYSTEM 455 CLEAN DATA VALIDATION

Purpose:
    Validate data/processed/system455_clean.csv before
    feature engineering and machine-learning modelling.

This script DOES NOT:
    - modify the dataset
    - delete observations
    - interpolate missing values
    - replace anomalies
    - add weather data

It only checks:
    1. File integrity
    2. Required columns
    3. Timestamp validity and ordering
    4. Sampling intervals and gaps
    5. Missing values
    6. Duplicate timestamps
    7. Sensor ranges
    8. Quality flags
    9. Target availability
   10. Potential target leakage
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "system455_clean.csv"
)


# =============================================================================
# EXPECTED COLUMNS
# =============================================================================

CORE_COLUMNS = [
    "timestamp",

    "collector_temp",
    "buffer_bottom",
    "buffer_centre",
    "buffer_top",

    "solar_radiation",
    "ambient_temp",

    "solar_energy",
    "solar_flow",
    "solar_power",

    "solar_temp_hot",
    "solar_temp_cold",

    "hotwater_energy",
    "hotwater_volume",
    "hotwater_flow",
    "hotwater_power",

    "hotwater_temp_hot",
    "hotwater_temp_cold",

    "pwm",
    "solar_pump",
    "electric_heater",

    "hour",
    "minute",
    "day_of_year",
    "day_of_week",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",

    "flag_buffer_bottom_below_0",
    "flag_collector_extreme",
    "flag_buffer_bottom_extreme",
    "flag_buffer_centre_extreme",
    "flag_buffer_top_extreme",

    "target_buffer_top_5min",
    "target_buffer_top_15min",
    "target_buffer_top_30min",
    "target_buffer_top_60min",
]


TEMPERATURE_COLUMNS = [
    "collector_temp",
    "buffer_bottom",
    "buffer_centre",
    "buffer_top",
    "ambient_temp",
    "solar_temp_hot",
    "solar_temp_cold",
    "hotwater_temp_hot",
    "hotwater_temp_cold",
]


FLOW_COLUMNS = [
    "solar_flow",
    "hotwater_flow",
]


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STMaCS SYSTEM 455 CLEAN DATA VALIDATION")
print("=" * 80)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading cleaned dataset...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Run clean_data.py first."
    )


df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# =============================================================================
# 1. COLUMN VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("1. COLUMN VALIDATION")
print("=" * 80)

missing_columns = [
    col
    for col in CORE_COLUMNS
    if col not in df.columns
]

unexpected_columns = [
    col
    for col in df.columns
    if col not in CORE_COLUMNS
]

if missing_columns:

    print("\n[FAIL] Missing required columns:")

    for col in missing_columns:
        print(f"  - {col}")

else:

    print("[OK] All expected columns are present.")


print(f"\nExpected columns: {len(CORE_COLUMNS)}")
print(f"Actual columns:   {len(df.columns)}")

if unexpected_columns:

    print("\nAdditional columns present:")

    for col in unexpected_columns:
        print(f"  - {col}")

else:

    print("\nNo unexpected columns.")


# =============================================================================
# 2. TIMESTAMP VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("2. TIMESTAMP VALIDATION")
print("=" * 80)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(
    f"Invalid timestamps: {invalid_timestamps:,}"
)

if invalid_timestamps == 0:
    print("[OK] All timestamps are valid.")
else:
    print("[FAIL] Invalid timestamps detected.")


# -----------------------------------------------------------------------------
# Check chronological ordering
# -----------------------------------------------------------------------------

is_sorted = df["timestamp"].is_monotonic_increasing

print(
    f"Chronologically sorted: "
    f"{'YES' if is_sorted else 'NO'}"
)

if is_sorted:
    print("[OK] Timestamp ordering is correct.")
else:
    print("[FAIL] Dataset is not chronologically sorted.")


# -----------------------------------------------------------------------------
# Duplicate timestamps
# -----------------------------------------------------------------------------

duplicate_timestamps = (
    df["timestamp"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate timestamps: {duplicate_timestamps:,}"
)

if duplicate_timestamps == 0:
    print("[OK] No duplicate timestamps.")
else:
    print("[WARNING] Duplicate timestamps detected.")


# =============================================================================
# 3. DATASET TIME SPAN
# =============================================================================

print("\n" + "=" * 80)
print("3. DATASET TIME SPAN")
print("=" * 80)

first_timestamp = df["timestamp"].min()
last_timestamp = df["timestamp"].max()

duration = last_timestamp - first_timestamp

print(f"First timestamp: {first_timestamp}")
print(f"Last timestamp:  {last_timestamp}")
print(f"Total duration:  {duration}")
print(f"Total days:      {duration.total_seconds() / 86400:.2f}")


# =============================================================================
# 4. SAMPLING INTERVAL
# =============================================================================

print("\n" + "=" * 80)
print("4. SAMPLING INTERVAL")
print("=" * 80)

time_diff = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

valid_intervals = time_diff.dropna()

if len(valid_intervals) > 0:

    print(
        f"Median interval:  "
        f"{valid_intervals.median():.2f} seconds"
    )

    print(
        f"Mean interval:    "
        f"{valid_intervals.mean():.2f} seconds"
    )

    print(
        f"Minimum interval: "
        f"{valid_intervals.min():.2f} seconds"
    )

    print(
        f"Maximum interval: "
        f"{valid_intervals.max():.2f} seconds"
    )

    dominant_interval = (
        valid_intervals
        .round()
        .value_counts()
        .idxmax()
    )

    dominant_count = (
        valid_intervals
        .round()
        .value_counts()
        .max()
    )

    dominant_percentage = (
        100
        * dominant_count
        / len(valid_intervals)
    )

    print(
        f"\nDominant interval: "
        f"{dominant_interval:.0f} seconds"
    )

    print(
        f"Dominant interval share: "
        f"{dominant_percentage:.2f}%"
    )


# =============================================================================
# 5. GAP ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("5. SAMPLING GAP ANALYSIS")
print("=" * 80)

# Normal sampling is approximately 5 minutes.
# We flag gaps greater than 10 minutes.

GAP_THRESHOLD_SECONDS = 10 * 60

gaps = df.loc[
    time_diff > GAP_THRESHOLD_SECONDS,
    ["timestamp"]
].copy()

gaps["previous_timestamp"] = (
    df["timestamp"]
    .shift(1)
    .loc[gaps.index]
)

gaps["gap_seconds"] = (
    time_diff.loc[gaps.index]
)

print(
    f"Potential gaps > 10 minutes: "
    f"{len(gaps):,}"
)

if len(gaps) > 0:

    largest_gap = gaps["gap_seconds"].max()

    print(
        f"Largest gap: "
        f"{largest_gap / 3600:.2f} hours"
    )

    print("\nLargest gaps:")

    largest = (
        gaps
        .sort_values("gap_seconds", ascending=False)
        .head(10)
    )

    for _, row in largest.iterrows():

        print(
            f"  {row['previous_timestamp']} → "
            f"{row['timestamp']} | "
            f"{row['gap_seconds'] / 60:.2f} minutes"
        )

else:

    print("[OK] No gaps larger than 10 minutes.")


# =============================================================================
# 6. MISSING VALUES
# =============================================================================

print("\n" + "=" * 80)
print("6. MISSING VALUE VALIDATION")
print("=" * 80)

missing_summary = []

for column in df.columns:

    missing = df[column].isna().sum()

    percentage = (
        100 * missing / len(df)
    )

    if missing > 0:

        missing_summary.append(
            (
                column,
                missing,
                percentage
            )
        )


if missing_summary:

    print(
        f"Columns containing missing values: "
        f"{len(missing_summary)}"
    )

    print("\nColumn                     Missing       Complete")
    print("-" * 60)

    for column, missing, percentage in missing_summary:

        complete = 100 - percentage

        print(
            f"{column:<28}"
            f"{missing:>8,}"
            f"       {complete:>7.2f}%"
        )

else:

    print("[OK] No missing values in the dataset.")


# =============================================================================
# 7. CORE ML VARIABLE COMPLETENESS
# =============================================================================

print("\n" + "=" * 80)
print("7. CORE ML VARIABLE COMPLETENESS")
print("=" * 80)

ml_core = [
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",
    "solar_radiation",
    "solar_flow",
    "solar_power",
    "hotwater_flow",
    "hotwater_power",
    "solar_pump",
    "electric_heater",
]


print(
    f"{'Variable':<25}"
    f"{'Missing':>12}"
    f"{'Complete':>14}"
)

print("-" * 55)

for column in ml_core:

    missing = df[column].isna().sum()

    complete = (
        100
        * (1 - missing / len(df))
    )

    print(
        f"{column:<25}"
        f"{missing:>12,}"
        f"{complete:>13.2f}%"
    )


# =============================================================================
# 8. SENSOR RANGE CHECK
# =============================================================================

print("\n" + "=" * 80)
print("8. SENSOR RANGE VALIDATION")
print("=" * 80)


range_limits = {

    "buffer_bottom": (0, 100),

    "buffer_centre": (0, 100),

    "buffer_top": (0, 100),

    "collector_temp": (-20, 180),

    "solar_radiation": (0, None),

    "solar_flow": (0, None),

    "hotwater_flow": (0, None),

    "solar_pump": (0, 1),

    "electric_heater": (0, 1),
}


range_flags = {}

for column, (lower, upper) in range_limits.items():

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    flag = pd.Series(
        False,
        index=df.index
    )

    if lower is not None:

        flag |= series < lower

    if upper is not None:

        flag |= series > upper

    count = flag.sum()

    range_flags[column] = count

    status = "OK" if count == 0 else "FLAGGED"

    print(
        f"{status:<8} "
        f"{column:<25} "
        f"{count:,} observations"
    )


# =============================================================================
# 9. QUALITY FLAGS CREATED DURING CLEANING
# =============================================================================

print("\n" + "=" * 80)
print("9. QUALITY FLAG VALIDATION")
print("=" * 80)


quality_columns = [
    "flag_buffer_bottom_below_0",
    "flag_collector_extreme",
    "flag_buffer_bottom_extreme",
    "flag_buffer_centre_extreme",
    "flag_buffer_top_extreme",
]


for column in quality_columns:

    if column not in df.columns:

        print(
            f"[MISSING] {column}"
        )

        continue

    count = df[column].fillna(False).astype(bool).sum()

    print(
        f"{column:<35}"
        f"{count:>8,} flagged"
    )


# =============================================================================
# 10. TARGET VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("10. FUTURE TARGET VALIDATION")
print("=" * 80)


target_columns = [
    "target_buffer_top_5min",
    "target_buffer_top_15min",
    "target_buffer_top_30min",
    "target_buffer_top_60min",
]


for target in target_columns:

    if target not in df.columns:

        print(
            f"[MISSING] {target}"
        )

        continue

    valid = df[target].notna().sum()

    missing = df[target].isna().sum()

    percentage = (
        100 * valid / len(df)
    )

    print(
        f"{target:<30}"
        f"Valid: {valid:>8,}"
        f"   Missing: {missing:>8,}"
        f"   Valid %: {percentage:>7.2f}%"
    )


# =============================================================================
# 11. TARGET RANGE
# =============================================================================

print("\n" + "=" * 80)
print("11. TARGET RANGE VALIDATION")
print("=" * 80)

for target in target_columns:

    if target not in df.columns:
        continue

    series = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    print(
        f"{target:<30}"
        f"Min: {series.min():>8.2f}"
        f"   Median: {series.median():>8.2f}"
        f"   Max: {series.max():>8.2f}"
    )


# =============================================================================
# 12. POTENTIAL TARGET LEAKAGE CHECK
# =============================================================================

print("\n" + "=" * 80)
print("12. TARGET LEAKAGE CHECK")
print("=" * 80)

print(
    """
The future Buffer Top Temperature targets should represent
temperature at a future point in time.

The current Buffer Top Temperature may be used as an input
feature because it represents the system's current state.

Future target columns themselves must NOT be used as predictors.
"""
)

for target in target_columns:

    if target in df.columns:

        print(
            f"[OK] Target exists: {target}"
        )


# =============================================================================
# 13. THERMAL STRATIFICATION CHECK
# =============================================================================

print("\n" + "=" * 80)
print("13. THERMAL STRATIFICATION CHECK")
print("=" * 80)

top_less_centre = (
    df["buffer_top"] < df["buffer_centre"]
).sum()

centre_less_bottom = (
    df["buffer_centre"] < df["buffer_bottom"]
).sum()

top_less_bottom = (
    df["buffer_top"] < df["buffer_bottom"]
).sum()


print(
    f"Top < Centre:     {top_less_centre:,}"
)

print(
    f"Centre < Bottom:  {centre_less_bottom:,}"
)

print(
    f"Top < Bottom:     {top_less_bottom:,}"
)

print(
    "\nThese are diagnostic indicators only."
)

print(
    "They are NOT automatically treated as invalid measurements."
)


# =============================================================================
# 14. TARGET VARIABILITY
# =============================================================================

print("\n" + "=" * 80)
print("14. TARGET VARIABILITY")
print("=" * 80)


for target in target_columns:

    if target not in df.columns:
        continue

    delta = (
        df[target]
        - df["buffer_top"]
    )

    delta = delta.dropna()

    if len(delta) == 0:
        continue

    print(
        f"\n{target}"
    )

    print(
        f"  Mean ΔT:       {delta.mean():.4f} °C"
    )

    print(
        f"  Std ΔT:        {delta.std():.4f} °C"
    )

    print(
        f"  Mean |ΔT|:     {delta.abs().mean():.4f} °C"
    )

    print(
        f"  Maximum |ΔT|:  {delta.abs().max():.4f} °C"
    )


# =============================================================================
# 15. FINAL VALIDATION STATUS
# =============================================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION STATUS")
print("=" * 80)


checks = {

    "File loaded successfully":
        True,

    "Required columns present":
        len(missing_columns) == 0,

    "Valid timestamps":
        invalid_timestamps == 0,

    "Chronological ordering":
        is_sorted,

    "No duplicate timestamps":
        duplicate_timestamps == 0,

    "Core buffer temperature complete":
        df["buffer_top"].notna().all()
        and df["buffer_centre"].notna().all()
        and df["buffer_bottom"].notna().all(),

    "Collector temperature complete":
        df["collector_temp"].notna().all(),

    "Solar radiation complete":
        df["solar_radiation"].notna().all(),

    "Future targets present":
        all(
            target in df.columns
            for target in target_columns
        ),
}


print()

all_passed = True

for check, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"{status:<6} {check}"
    )

    if not passed:
        all_passed = False


# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

if all_passed:

    print(
        """
SYSTEM 455 PASSED THE CORE DATA VALIDATION.

The dataset is structurally suitable for the next stage:

    1. Feature engineering
    2. Actual-time-based target construction
    3. Chronological train/test splitting
    4. Baseline model development

Important:
    Validation does not mean every measurement is physically perfect.

Known issues such as:
    - 94 buffer-bottom values below 0°C
    - missing ambient temperature
    - sampling gaps
    - thermal stratification violations

remain documented and are not silently removed.
"""
    )

else:

    print(
        """
SYSTEM 455 DID NOT PASS ALL VALIDATION CHECKS.

Review the FAIL items above before proceeding
to model development.

No data has been modified.
"""
    )


print("=" * 80)

