"""
STMaCS SYSTEM 455
FEATURE ENGINEERING AND TIME-BASED TARGET CONSTRUCTION

Purpose:
    Prepare the validated System 455 dataset for machine-learning.

This script:
    1. Loads the validated System 455 dataset.
    2. Reconstructs future targets using ACTUAL timestamps.
    3. Creates physically meaningful thermal features.
    4. Creates cyclic time features.
    5. Creates lag/rate-of-change features.
    6. Preserves quality flags.
    7. Prevents future-data leakage.
    8. Saves an ML-ready dataset.

IMPORTANT:
    No raw measurements are overwritten.
    No suspicious measurements are silently deleted.
    Future target values are NOT used as input features.
"""

from pathlib import Path
import numpy as np
import pandas as pd


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "system455_features.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_COLUMN = "buffer_top"

HORIZONS = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_column(df, name):
    """
    Return a column if it exists, otherwise raise a clear error.
    """
    if name not in df.columns:
        raise ValueError(
            f"Required column '{name}' was not found in the dataset."
        )
    return name


# =============================================================================
# LOAD DATA
# =============================================================================

section("STMaCS SYSTEM 455 FEATURE ENGINEERING")

print("Loading validated System 455 dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Run validate_clean_data.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df):,}")


# =============================================================================
# TIMESTAMP
# =============================================================================

section("TIMESTAMP PREPARATION")

find_column(df, "timestamp")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid}")

if invalid > 0:
    raise ValueError(
        "Invalid timestamps detected. Fix the dataset before feature engineering."
    )

df = df.sort_values("timestamp").reset_index(drop=True)

duplicates = df["timestamp"].duplicated().sum()

print(f"Duplicate timestamps: {duplicates}")

if duplicates > 0:
    raise ValueError(
        "Duplicate timestamps detected. Resolve them before target construction."
    )


# =============================================================================
# IDENTIFY CORE VARIABLES
# =============================================================================

section("IDENTIFYING CORE SYSTEM 455 VARIABLES")

required_columns = [
    "collector_temp",
    "buffer_bottom",
    "buffer_centre",
    "buffer_top",
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
]

for column in required_columns:
    find_column(df, column)
    print(f"[FOUND] {column}")


# =============================================================================
# CURRENT STATE FEATURES
# =============================================================================

section("CREATING CURRENT-STATE FEATURES")

# Thermal state
df["thermal_stratification_top_centre"] = (
    df["buffer_top"] - df["buffer_centre"]
)

df["thermal_stratification_centre_bottom"] = (
    df["buffer_centre"] - df["buffer_bottom"]
)

df["thermal_stratification_top_bottom"] = (
    df["buffer_top"] - df["buffer_bottom"]
)

# Collector/storage relationship
df["collector_minus_buffer_top"] = (
    df["collector_temp"] - df["buffer_top"]
)

df["collector_minus_buffer_centre"] = (
    df["collector_temp"] - df["buffer_centre"]
)

df["collector_minus_buffer_bottom"] = (
    df["collector_temp"] - df["buffer_bottom"]
)

# Heat-transfer loop temperature difference
df["solar_loop_delta_t"] = (
    df["solar_temp_hot"] - df["solar_temp_cold"]
)

# Hot-water temperature difference
df["hotwater_delta_t"] = (
    df["hotwater_temp_hot"] - df["hotwater_temp_cold"]
)

print("Created thermal state and temperature-difference features.")


# =============================================================================
# TIME FEATURES
# =============================================================================

section("CREATING CYCLIC TIME FEATURES")

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Decimal hour gives a better representation of the actual time of day.
df["hour_decimal"] = (
    df["timestamp"].dt.hour
    + df["timestamp"].dt.minute / 60.0
    + df["timestamp"].dt.second / 3600.0
)

# Cyclic encoding of time of day.
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour_decimal"] / 24.0
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour_decimal"] / 24.0
)

# Cyclic encoding of day of week.
df["day_of_week_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7.0
)

df["day_of_week_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7.0
)

# Cyclic encoding of annual seasonality.
df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

print("Created:")
print("  - hour")
print("  - minute")
print("  - day_of_week")
print("  - day_of_year")
print("  - hour_sin / hour_cos")
print("  - day_of_week_sin / day_of_week_cos")
print("  - day_of_year_sin / day_of_year_cos")


# =============================================================================
# LAG FEATURES
# =============================================================================

section("CREATING LAG FEATURES")

# Because the dominant sampling interval is 5 minutes,
# these lags represent approximately:
#
# lag_1  = 5 minutes
# lag_3  = 15 minutes
# lag_6  = 30 minutes
# lag_12 = 60 minutes

lag_variables = [
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",
    "solar_radiation",
    "solar_flow",
    "hotwater_flow",
]

lag_steps = [1, 3, 6, 12]

for variable in lag_variables:

    for lag in lag_steps:

        new_name = f"{variable}_lag_{lag}"

        df[new_name] = df[variable].shift(lag)

print("Created lag features for:")
for variable in lag_variables:
    print(f"  - {variable}")


# =============================================================================
# RATE OF CHANGE FEATURES
# =============================================================================

section("CREATING THERMAL RATE-OF-CHANGE FEATURES")

# Calculate changes over approximately:
# 5, 15, 30 and 60 minutes.

for variable in [
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",
]:

    for lag in [1, 3, 6, 12]:

        new_name = f"{variable}_delta_{lag}"

        df[new_name] = (
            df[variable]
            - df[variable].shift(lag)
        )

print("Created temperature-change features.")


# =============================================================================
# ACTUAL-TIME FUTURE TARGETS
# =============================================================================

section("CREATING ACTUAL-TIME FUTURE TARGETS")

print(
    "\nTargets will be created using timestamps rather than simply "
    "assuming every row represents exactly five minutes."
)

# Keep timestamp separate for target lookup.
target_reference = df[
    ["timestamp", TARGET_COLUMN]
].copy()

target_reference = target_reference.rename(
    columns={TARGET_COLUMN: "future_value"}
)

# Use timestamp as index for efficient time-based lookup.
target_reference = target_reference.set_index("timestamp")

for label, minutes in HORIZONS.items():

    target_name = f"target_buffer_top_{label}"

    print(f"\nCreating {minutes}-minute target...")

    future_times = (
        df["timestamp"]
        + pd.Timedelta(minutes=minutes)
    )

    # Find the closest actual recorded measurement to the requested
    # future time, subject to a strict tolerance.
    #
    # We allow up to 2 minutes around the requested time.
    #
    # This prevents a 60-minute target from accidentally matching
    # a measurement several hours later across a logging gap.

    lookup = pd.DataFrame(
        {
            "future_time": future_times,
            "row_id": np.arange(len(df)),
        }
    )

    lookup = lookup.sort_values("future_time")

    source = target_reference.reset_index().sort_values("timestamp")

    matched = pd.merge_asof(
        lookup,
        source,
        left_on="future_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=2),
    )

    matched = matched.sort_values("row_id")

    df[target_name] = matched["future_value"].to_numpy()

    valid = df[target_name].notna().sum()
    missing = df[target_name].isna().sum()

    print(
        f"  Valid targets:   {valid:,}"
    )

    print(
        f"  Missing targets: {missing:,}"
    )


# =============================================================================
# TARGET CHANGE FEATURES
# =============================================================================

section("CREATING FUTURE TEMPERATURE CHANGE TARGETS")

for label in HORIZONS:

    target_name = f"target_buffer_top_{label}"

    delta_name = f"target_delta_buffer_top_{label}"

    df[delta_name] = (
        df[target_name] - df["buffer_top"]
    )

    print(
        f"[CREATED] {delta_name}"
    )


# =============================================================================
# QUALITY FLAGS
# =============================================================================

section("CHECKING QUALITY FLAGS")

quality_columns = [
    column
    for column in df.columns
    if column.startswith("flag_")
]

print(f"Quality flag columns found: {len(quality_columns)}")

for column in quality_columns:
    print(
        f"  {column}: {int(df[column].sum()):,} flagged"
    )


# =============================================================================
# DEFINE ML FEATURE SET
# =============================================================================

section("DEFINING MACHINE-LEARNING FEATURES")

feature_columns = [

    # -------------------------------------------------------------------------
    # Current thermal state
    # -------------------------------------------------------------------------
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",

    # -------------------------------------------------------------------------
    # Solar conditions
    # -------------------------------------------------------------------------
    "solar_radiation",
    "solar_flow",
    "solar_power",
    "solar_temp_hot",
    "solar_temp_cold",

    # -------------------------------------------------------------------------
    # Hot-water demand
    # -------------------------------------------------------------------------
    "hotwater_flow",
    "hotwater_power",
    "hotwater_temp_hot",
    "hotwater_temp_cold",

    # -------------------------------------------------------------------------
    # Control states
    # -------------------------------------------------------------------------
    "pwm",
    "solar_pump",
    "electric_heater",

    # -------------------------------------------------------------------------
    # Environmental condition
    # -------------------------------------------------------------------------
    "ambient_temp",

    # -------------------------------------------------------------------------
    # Thermal relationships
    # -------------------------------------------------------------------------
    "thermal_stratification_top_centre",
    "thermal_stratification_centre_bottom",
    "thermal_stratification_top_bottom",

    "collector_minus_buffer_top",
    "collector_minus_buffer_centre",
    "collector_minus_buffer_bottom",

    "solar_loop_delta_t",
    "hotwater_delta_t",

    # -------------------------------------------------------------------------
    # Time features
    # -------------------------------------------------------------------------
    "hour",
    "day_of_week",
    "day_of_year",

    "hour_sin",
    "hour_cos",

    "day_of_week_sin",
    "day_of_week_cos",

    "day_of_year_sin",
    "day_of_year_cos",

    # -------------------------------------------------------------------------
    # Lag features
    # -------------------------------------------------------------------------

    "buffer_top_lag_1",
    "buffer_top_lag_3",
    "buffer_top_lag_6",
    "buffer_top_lag_12",

    "buffer_centre_lag_1",
    "buffer_centre_lag_3",
    "buffer_centre_lag_6",
    "buffer_centre_lag_12",

    "buffer_bottom_lag_1",
    "buffer_bottom_lag_3",
    "buffer_bottom_lag_6",
    "buffer_bottom_lag_12",

    "collector_temp_lag_1",
    "collector_temp_lag_3",
    "collector_temp_lag_6",
    "collector_temp_lag_12",

    "solar_radiation_lag_1",
    "solar_radiation_lag_3",
    "solar_radiation_lag_6",
    "solar_radiation_lag_12",

    "solar_flow_lag_1",
    "solar_flow_lag_3",
    "solar_flow_lag_6",
    "solar_flow_lag_12",

    "hotwater_flow_lag_1",
    "hotwater_flow_lag_3",
    "hotwater_flow_lag_6",
    "hotwater_flow_lag_12",

    # -------------------------------------------------------------------------
    # Temperature-change features
    # -------------------------------------------------------------------------

    "buffer_top_delta_1",
    "buffer_top_delta_3",
    "buffer_top_delta_6",
    "buffer_top_delta_12",

    "buffer_centre_delta_1",
    "buffer_centre_delta_3",
    "buffer_centre_delta_6",
    "buffer_centre_delta_12",

    "buffer_bottom_delta_1",
    "buffer_bottom_delta_3",
    "buffer_bottom_delta_6",
    "buffer_bottom_delta_12",

    "collector_temp_delta_1",
    "collector_temp_delta_3",
    "collector_temp_delta_6",
    "collector_temp_delta_12",
]


# Remove duplicates while preserving order.
feature_columns = list(dict.fromkeys(feature_columns))


missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    print("\nMissing feature columns:")

    for column in missing_features:
        print(f"  [MISSING] {column}")

    raise ValueError(
        "One or more expected ML feature columns are missing."
    )

print(
    f"\nTotal engineered feature columns: {len(feature_columns)}"
)


# =============================================================================
# TARGET COLUMNS
# =============================================================================

target_columns = [
    f"target_buffer_top_{label}"
    for label in HORIZONS
]

target_delta_columns = [
    f"target_delta_buffer_top_{label}"
    for label in HORIZONS
]


# =============================================================================
# QUALITY / IDENTIFICATION COLUMNS
# =============================================================================

quality_columns = [
    column
    for column in df.columns
    if column.startswith("flag_")
]

output_columns = (
    [
        "timestamp"
    ]
    + feature_columns
    + target_columns
    + target_delta_columns
    + quality_columns
)


# Remove duplicate column names.
output_columns = list(dict.fromkeys(output_columns))


# =============================================================================
# BUILD FINAL DATASET
# =============================================================================

section("BUILDING FEATURE DATASET")

ml_df = df[output_columns].copy()

print(f"Rows:    {len(ml_df):,}")
print(f"Columns: {len(ml_df.columns):,}")


# =============================================================================
# FEATURE COMPLETENESS
# =============================================================================

section("FEATURE COMPLETENESS")

print(
    f"{'Feature':40s}"
    f"{'Missing':>12s}"
    f"{'Complete %':>14s}"
)

print("-" * 68)

for column in feature_columns:

    missing = ml_df[column].isna().sum()

    complete = (
        100
        * (len(ml_df) - missing)
        / len(ml_df)
    )

    print(
        f"{column:40s}"
        f"{missing:12,d}"
        f"{complete:13.2f}%"
    )


# =============================================================================
# TARGET COMPLETENESS
# =============================================================================

section("TARGET COMPLETENESS")

for target in target_columns:

    valid = ml_df[target].notna().sum()

    missing = ml_df[target].isna().sum()

    percentage = (
        100 * valid / len(ml_df)
    )

    print(
        f"{target:30s}"
        f"Valid: {valid:8,d}"
        f"Missing: {missing:6,d}"
        f"({percentage:6.2f}%)"
    )


# =============================================================================
# IMPORTANT: DO NOT DROP ROWS HERE
# =============================================================================

print(
    "\nNo rows are being dropped at this stage."
)

print(
    "Rows with missing targets will be handled when constructing "
    "each individual train/test dataset."
)

print(
    "This allows the same engineered dataset to support multiple "
    "prediction horizons."
)


# =============================================================================
# SAVE
# =============================================================================

section("SAVING FEATURE DATASET")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nFeature-engineered dataset saved to:\n"
    f"  {OUTPUT_FILE}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("FEATURE ENGINEERING COMPLETE")

print(f"Input rows:          {len(df):,}")
print(f"Output rows:         {len(ml_df):,}")
print(f"Output columns:      {len(ml_df.columns):,}")

print(
    "\nPrediction targets:"
)

for label, minutes in HORIZONS.items():

    target = f"target_buffer_top_{label}"

    valid = ml_df[target].notna().sum()

    print(
        f"  {minutes:2d}-minute future Buffer Top: "
        f"{valid:,} valid samples"
    )

print(
    "\nImportant:"
)

print(
    "  - Actual timestamps were used for future target construction."
)

print(
    "  - Current Buffer Top is allowed as a predictor."
)

print(
    "  - Future target columns are NOT predictors."
)

print(
    "  - Quality flags are preserved."
)

print(
    "  - No suspicious measurements were automatically deleted."
)

print(
    "  - No weather data has been inserted."
)

print(
    "  - No interpolation has been performed."
)

print(
    "\nNext stage:"
)

print(
    "  1. Examine feature-target relationships."
)

print(
    "  2. Decide whether ambient temperature needs treatment."
)

print(
    "  3. Build chronological train/test datasets."
)

print(
    "  4. Train a baseline model."
)

print(
    "  5. Compare 5, 15, 30 and 60-minute prediction performance."
)

print(
    "\n" + "=" * 80
)
print("READY FOR BASELINE MODEL DEVELOPMENT")
print("=" * 80)