"""
STMaCS SYSTEM 455 MACHINE-LEARNING DATASET BUILDER

Purpose:
    Build clean, horizon-specific ML datasets from the feature-engineered
    System 455 dataset.

Important methodology:
    - Chronological train/test split
    - No random shuffling
    - No future target leakage
    - Correct known sensor sentinel values
    - Do NOT impute a completely invalid sensor channel
    - Ambient temperature is excluded because the entire channel is invalid
    - Future targets are retained only as targets
    - Training samples whose actual future targets cross into the test period
      are excluded
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
    / "system455_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FRACTION = 0.80

HORIZONS = {
    "5min": "target_buffer_top_5min",
    "15min": "target_buffer_top_15min",
    "30min": "target_buffer_top_30min",
    "60min": "target_buffer_top_60min",
}

HORIZON_MINUTES = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}

# Known invalid sentinel in the ambient temperature channel
AMBIENT_SENTINEL = 3276.7

# Physically reasonable ambient temperature limits for validation
AMBIENT_MIN = -50.0
AMBIENT_MAX = 70.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


# =============================================================================
# LOAD DATA
# =============================================================================

print_section("STMaCS SYSTEM 455 MACHINE-LEARNING DATASET BUILDER")

print_section("LOADING FEATURE-ENGINEERED DATASET")

print("Input file:")
print(f"  {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# =============================================================================
# TIMESTAMP PREPARATION
# =============================================================================

print_section("TIMESTAMP PREPARATION")

if "timestamp" not in df.columns:
    raise ValueError("The dataset must contain a 'timestamp' column.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError(
        f"Found {invalid_timestamps:,} invalid timestamps."
    )

duplicate_timestamps = df["timestamp"].duplicated().sum()

print(f"Duplicate timestamps: {duplicate_timestamps}")

if duplicate_timestamps > 0:
    raise ValueError(
        f"Found {duplicate_timestamps:,} duplicate timestamps."
    )

df = df.sort_values("timestamp").reset_index(drop=True)

print("[OK] Timestamp validation passed.")


# =============================================================================
# DEFINE CORE PREDICTORS
# =============================================================================

print_section("DEFINING FINAL PREDICTOR SET")

current_state_features = [
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",
    "solar_radiation",
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

thermal_difference_features = [
    "thermal_stratification_top_centre",
    "thermal_stratification_centre_bottom",
    "thermal_stratification_top_bottom",
    "collector_minus_buffer_top",
    "collector_minus_buffer_centre",
    "collector_minus_buffer_bottom",
    "solar_loop_delta_t",
    "hotwater_delta_t",
]

time_features = [
    "hour",
    "day_of_week",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

lag_features = [
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
]

rate_of_change_features = [
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

    "solar_radiation_delta_1",
    "solar_radiation_delta_3",
]

rate_of_change_features = [
    col for col in rate_of_change_features
    if col in df.columns
]

predictor_features = (
    current_state_features
    + thermal_difference_features
    + time_features
    + lag_features
    + rate_of_change_features
)

predictor_features = list(
    dict.fromkeys(predictor_features)
)

print(f"Current-state features:       {len(current_state_features)}")
print(f"Thermal-difference features:  {len(thermal_difference_features)}")
print(f"Time features:                {len(time_features)}")
print(f"Lag features:                 {len(lag_features)}")
print(f"Rate-of-change features:      {len(rate_of_change_features)}")
print()
print(f"Total predictor features:     {len(predictor_features)}")


# =============================================================================
# VERIFY PREDICTORS
# =============================================================================

print_section("VERIFYING PREDICTOR FEATURES")

require_columns(df, predictor_features)

print("[OK] All predictor columns are present.")


# =============================================================================
# CORRECT AMBIENT TEMPERATURE SENTINELS
# =============================================================================

print_section("CORRECTING AMBIENT TEMPERATURE SENTINELS")

if "ambient_temp" not in df.columns:

    print("[INFO] Ambient temperature column does not exist.")

else:

    print("Known invalid ambient-temperature sentinel values:")
    print(f"  {AMBIENT_SENTINEL} °C")

    sentinel_mask = df["ambient_temp"].eq(
        AMBIENT_SENTINEL
    )

    sentinel_count = sentinel_mask.sum()

    print(
        f"[FOUND] {sentinel_count:,} observations equal to "
        f"{AMBIENT_SENTINEL} °C"
    )

    df.loc[
        sentinel_mask,
        "ambient_temp"
    ] = np.nan

    print()
    print(
        f"Total sentinel observations converted to NaN: "
        f"{sentinel_count:,}"
    )


# =============================================================================
# AMBIENT TEMPERATURE VALIDATION
# =============================================================================

print_section("AMBIENT TEMPERATURE RANGE CHECK")

if "ambient_temp" in df.columns:

    below_min = (
        df["ambient_temp"] < AMBIENT_MIN
    ).sum()

    above_max = (
        df["ambient_temp"] > AMBIENT_MAX
    ).sum()

    print(f"Below {AMBIENT_MIN} °C: {below_min:,}")
    print(f"Above {AMBIENT_MAX} °C: {above_max:,}")

    invalid_range_mask = (
        (df["ambient_temp"] < AMBIENT_MIN)
        | (df["ambient_temp"] > AMBIENT_MAX)
    )

    invalid_range_count = invalid_range_mask.sum()

    if invalid_range_count > 0:

        print(
            f"[CORRECTED] {invalid_range_count:,} "
            "out-of-range ambient values converted to NaN."
        )

        df.loc[
            invalid_range_mask,
            "ambient_temp"
        ] = np.nan


# =============================================================================
# AMBIENT TEMPERATURE STATUS
# =============================================================================

print_section("AMBIENT TEMPERATURE STATUS")

if "ambient_temp" in df.columns:

    ambient_missing = df["ambient_temp"].isna().sum()
    ambient_valid = df["ambient_temp"].notna().sum()

    print(
        f"Ambient temperature missing:   {ambient_missing:,}"
    )

    print(
        f"Ambient temperature valid:     {ambient_valid:,}"
    )

    if ambient_valid == 0:

        print()
        print(
            "[WARNING] Ambient temperature has ZERO valid observations."
        )

        print()
        print("Ambient temperature will NOT be:")
        print("  - median imputed")
        print("  - forward filled")
        print("  - interpolated")
        print("  - replaced with synthetic values")

        print()
        print(
            "The ambient-temperature predictor is excluded from "
            "the ML datasets because the sensor channel contains "
            "no usable data."
        )


# =============================================================================
# VERIFY TARGETS
# =============================================================================

print_section("VERIFYING FUTURE TARGETS")

require_columns(
    df,
    list(HORIZONS.values())
)

print("[OK] All target columns are present.")

for horizon, target in HORIZONS.items():

    valid = df[target].notna().sum()
    missing = df[target].isna().sum()

    print(
        f"{target:<35}"
        f"Valid: {valid:>8,}   "
        f"Missing: {missing:>6,}"
    )


# =============================================================================
# TARGET LEAKAGE CHECK
# =============================================================================

print_section("TARGET LEAKAGE CHECK")

future_target_columns = [
    col
    for col in df.columns
    if col.startswith("target_")
]

leakage = set(
    predictor_features
).intersection(
    future_target_columns
)

if leakage:

    raise ValueError(
        "TARGET LEAKAGE DETECTED!\n"
        + "\n".join(
            f"  - {x}"
            for x in sorted(leakage)
        )
    )

print(
    "[OK] No future target columns are being used as predictors."
)


# =============================================================================
# CREATE CHRONOLOGICAL TRAIN / TEST SPLIT
# =============================================================================

print_section(
    "CREATING CHRONOLOGICAL TRAIN / TEST SPLIT"
)

total_rows = len(df)

train_rows = int(
    total_rows * TRAIN_FRACTION
)

train_df = df.iloc[
    :train_rows
].copy()

test_df = df.iloc[
    train_rows:
].copy()

train_end_timestamp = train_df[
    "timestamp"
].iloc[-1]

test_start_timestamp = test_df[
    "timestamp"
].iloc[0]

print(
    f"Total rows:       {total_rows:,}"
)

print(
    f"Training rows:    {len(train_df):,}"
)

print(
    f"Testing rows:     {len(test_df):,}"
)

print()
print("Training period:")
print(
    f"  {train_df['timestamp'].iloc[0]}"
)
print(
    f"  → {train_end_timestamp}"
)

print()
print("Testing period:")
print(
    f"  {test_start_timestamp}"
)
print(
    f"  → {test_df['timestamp'].iloc[-1]}"
)

boundary_gap = (
    test_start_timestamp
    - train_end_timestamp
)

print()
print(
    "Train/test boundary separation: "
    f"{boundary_gap.total_seconds() / 60:.2f} minutes"
)

print()
print("[OK] Chronological split created.")
print("[OK] No random shuffling performed.")


# =============================================================================
# REMOVE COMPLETELY INVALID AMBIENT TEMPERATURE
# =============================================================================

print_section(
    "FINAL AMBIENT TEMPERATURE DECISION"
)

if "ambient_temp" in predictor_features:

    ambient_valid = (
        df["ambient_temp"].notna().sum()
    )

    if ambient_valid == 0:

        predictor_features.remove(
            "ambient_temp"
        )

        print(
            "[REMOVED] ambient_temp from predictor set "
            "because zero valid observations remain."
        )

    else:

        raise ValueError(
            "Ambient temperature contains valid values but the "
            "script reached an unexpected decision point."
        )

if "ambient_temp_missing" in predictor_features:

    predictor_features.remove(
        "ambient_temp_missing"
    )

print()
print(
    f"Final predictor count: {len(predictor_features)}"
)

print()
print("Final predictor set contains:")

for feature in predictor_features:
    print(f"  - {feature}")


# =============================================================================
# SAVE PREDICTOR LIST
# =============================================================================

predictor_file = (
    OUTPUT_DIR
    / "predictor_features.txt"
)

with open(
    predictor_file,
    "w",
    encoding="utf-8"
) as f:

    for feature in predictor_features:
        f.write(
            feature + "\n"
        )

print()
print(
    f"[SAVED] {predictor_file}"
)


# =============================================================================
# VERIFY PREDICTOR COMPLETENESS
# =============================================================================

print_section(
    "VERIFYING PREDICTOR COMPLETENESS"
)

predictor_missing_counts = (
    df[predictor_features]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_predictors = (
    predictor_missing_counts[
        predictor_missing_counts > 0
    ]
)

if len(missing_predictors) == 0:

    print(
        "[OK] All final predictor columns are complete."
    )

else:

    print(
        "Predictors containing missing values:"
    )

    for feature, missing in missing_predictors.items():

        completeness = (
            (len(df) - missing)
            / len(df)
            * 100
        )

        print(
            f"  {feature:<40}"
            f"Missing: {missing:>8,} "
            f"Complete: {completeness:>7.2f}%"
        )


# =============================================================================
# BUILD HORIZON-SPECIFIC DATASETS
# =============================================================================

print_section(
    "BUILDING HORIZON-SPECIFIC DATASETS"
)

dataset_summary = []


for horizon, target in HORIZONS.items():

    horizon_minutes = HORIZON_MINUTES[
        horizon
    ]

    print()
    print("-" * 80)
    print(
        f"PROCESSING {horizon.upper()} "
        "PREDICTION HORIZON"
    )
    print("-" * 80)

    print("Target:")
    print(
        f"  {target}"
    )

    # -------------------------------------------------------------------------
    # Determine required columns
    # -------------------------------------------------------------------------

    required_columns = (
        ["timestamp"]
        + predictor_features
        + [target]
    )

    horizon_df = df[
        required_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Split using same chronological boundary
    # -------------------------------------------------------------------------

    horizon_train = horizon_df.iloc[
        :train_rows
    ].copy()

    horizon_test = horizon_df.iloc[
        train_rows:
    ].copy()

    original_training_rows = len(
        horizon_train
    )

    original_testing_rows = len(
        horizon_test
    )

    # -------------------------------------------------------------------------
    # Remove rows with missing target
    # -------------------------------------------------------------------------

    train_target_missing = (
        horizon_train[target]
        .isna()
        .sum()
    )

    test_target_missing = (
        horizon_test[target]
        .isna()
        .sum()
    )

    print()
    print(
        "Rows removed because target was unavailable:"
    )

    print(
        f"  Training: {train_target_missing:,}"
    )

    print(
        f"  Testing:  {test_target_missing:,}"
    )

    horizon_train = horizon_train.dropna(
        subset=[target]
    )

    horizon_test = horizon_test.dropna(
        subset=[target]
    )

    # -------------------------------------------------------------------------
    # Remove training samples whose future target crosses the test boundary
    # -------------------------------------------------------------------------

    print()
    print(
        "CHECKING FUTURE TARGET BOUNDARY ISOLATION"
    )

    # The feature-engineering stage constructs each target for an intended
    # future timestamp. The target-boundary audit confirmed that the actual
    # matched targets crossing into the test period occur at the final
    # horizon-dependent observations of the training period.
    #
    # To maintain strict chronological isolation, exclude training
    # observations whose intended future timestamp is at or after the
    # first testing timestamp.

    training_future_times = (
        horizon_train["timestamp"]
        + pd.Timedelta(
            minutes=horizon_minutes
        )
    )

    boundary_crossing_mask = (
        training_future_times
        >= test_start_timestamp
    )

    boundary_crossing_count = (
        boundary_crossing_mask.sum()
    )

    print(
        f"Prediction horizon: {horizon_minutes} minutes"
    )

    print(
        f"Test period begins:  {test_start_timestamp}"
    )

    print(
        f"Training targets crossing test boundary: "
        f"{boundary_crossing_count:,}"
    )

    if boundary_crossing_count > 0:

        first_crossing = (
            horizon_train.loc[
                boundary_crossing_mask,
                "timestamp"
            ].iloc[0]
        )

        last_crossing = (
            horizon_train.loc[
                boundary_crossing_mask,
                "timestamp"
            ].iloc[-1]
        )

        print()
        print(
            "Boundary-crossing training observations:"
        )

        print(
            f"  First: {first_crossing}"
        )

        print(
            f"  Last : {last_crossing}"
        )

        print()
        print(
            "[REMOVED] These observations are excluded "
            "to prevent future target leakage."
        )

        horizon_train = horizon_train.loc[
            ~boundary_crossing_mask
        ].copy()

    else:

        print(
            "[OK] No training targets cross into "
            "the test period."
        )

    # -------------------------------------------------------------------------
    # Predictor missingness
    # -------------------------------------------------------------------------

    train_predictor_missing = (
        horizon_train[
            predictor_features
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    test_predictor_missing = (
        horizon_test[
            predictor_features
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print()
    print(
        "Rows removed because predictor features "
        "were unavailable:"
    )

    print(
        f"  Training: {train_predictor_missing:,}"
    )

    print(
        f"  Testing:  {test_predictor_missing:,}"
    )

    horizon_train = horizon_train.dropna(
        subset=predictor_features
    )

    horizon_test = horizon_test.dropna(
        subset=predictor_features
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    remaining_train_missing = (
        horizon_train[
            predictor_features + [target]
        ]
        .isna()
        .sum()
        .sum()
    )

    remaining_test_missing = (
        horizon_test[
            predictor_features + [target]
        ]
        .isna()
        .sum()
        .sum()
    )

    if remaining_train_missing > 0:

        raise ValueError(
            f"Training dataset for {horizon} still contains "
            f"{remaining_train_missing} missing predictor/target values."
        )

    if remaining_test_missing > 0:

        raise ValueError(
            f"Testing dataset for {horizon} still contains "
            f"{remaining_test_missing} missing predictor/target values."
        )

    # -------------------------------------------------------------------------
    # Verify final temporal isolation
    # -------------------------------------------------------------------------

    final_train_max = (
        horizon_train["timestamp"].max()
    )

    if final_train_max >= test_start_timestamp:

        raise ValueError(
            f"Temporal isolation failed for {horizon}: "
            "training data extends into the test period."
        )

    print()
    print(
        "[OK] Training data remains entirely before "
        "the test period."
    )

    print(
        f"  Final training timestamp: {final_train_max}"
    )

    print(
        f"  First testing timestamp:  {test_start_timestamp}"
    )

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------

    final_columns = (
        ["timestamp"]
        + predictor_features
        + [target]
    )

    horizon_train = horizon_train[
        final_columns
    ]

    horizon_test = horizon_test[
        final_columns
    ]

    # -------------------------------------------------------------------------
    # Save datasets
    # -------------------------------------------------------------------------

    train_file = (
        OUTPUT_DIR
        / f"system455_{horizon}_train.csv"
    )

    test_file = (
        OUTPUT_DIR
        / f"system455_{horizon}_test.csv"
    )

    horizon_train.to_csv(
        train_file,
        index=False
    )

    horizon_test.to_csv(
        test_file,
        index=False
    )

    print()
    print("Dataset summary:")
    print(
        f"  Original training rows: "
        f"{original_training_rows:,}"
    )

    print(
        f"  Final training rows:    "
        f"{len(horizon_train):,}"
    )

    print(
        f"  Testing rows:           "
        f"{len(horizon_test):,}"
    )

    print(
        f"  Predictors:             "
        f"{len(predictor_features):,}"
    )

    print()
    print("Training period:")

    if len(horizon_train) > 0:

        print(
            f"  {horizon_train['timestamp'].iloc[0]}"
        )

        print(
            f"  → {horizon_train['timestamp'].iloc[-1]}"
        )

    print()
    print("Testing period:")

    if len(horizon_test) > 0:

        print(
            f"  {horizon_test['timestamp'].iloc[0]}"
        )

        print(
            f"  → {horizon_test['timestamp'].iloc[-1]}"
        )

    print()
    print(
        f"[SAVED] {train_file}"
    )

    print(
        f"[SAVED] {test_file}"
    )

    # -------------------------------------------------------------------------
    # Store summary
    # -------------------------------------------------------------------------

    dataset_summary.append({
        "horizon": horizon,
        "target": target,
        "original_training_rows": original_training_rows,
        "training_target_missing_removed": train_target_missing,
        "boundary_crossing_removed": boundary_crossing_count,
        "training_predictor_missing_removed": train_predictor_missing,
        "final_training_rows": len(horizon_train),
        "testing_target_missing_removed": test_target_missing,
        "testing_predictor_missing_removed": test_predictor_missing,
        "final_testing_rows": len(horizon_test),
        "predictor_count": len(predictor_features),
        "training_start": (
            horizon_train["timestamp"].iloc[0]
            if len(horizon_train) > 0
            else None
        ),
        "training_end": (
            horizon_train["timestamp"].iloc[-1]
            if len(horizon_train) > 0
            else None
        ),
        "testing_start": (
            horizon_test["timestamp"].iloc[0]
            if len(horizon_test) > 0
            else None
        ),
        "testing_end": (
            horizon_test["timestamp"].iloc[-1]
            if len(horizon_test) > 0
            else None
        ),
    })


# =============================================================================
# SAVE SUMMARY
# =============================================================================

print_section(
    "SAVING DATASET SUMMARY"
)

summary_df = pd.DataFrame(
    dataset_summary
)

summary_file = (
    OUTPUT_DIR
    / "dataset_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print(
    f"[SAVED] {summary_file}"
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print_section(
    "FINAL DATASET VALIDATION"
)

for _, row in summary_df.iterrows():

    print(
        f"[PASS] "
        f"{row['horizon']:>5} | "
        f"train={row['final_training_rows']:>7,} | "
        f"test={row['final_testing_rows']:>7,} | "
        f"features={row['predictor_count']:>3} | "
        f"boundary_removed="
        f"{row['boundary_crossing_removed']:>3}"
    )


# =============================================================================
# COMPLETION
# =============================================================================

print_section(
    "MACHINE-LEARNING DATASET BUILD COMPLETE"
)

print("Output directory:")
print(
    f"  {OUTPUT_DIR}"
)

print()
print("Files created:")

for horizon in HORIZONS:

    print(
        f"  - system455_{horizon}_train.csv"
    )

    print(
        f"  - system455_{horizon}_test.csv"
    )

print(
    "  - predictor_features.txt"
)

print(
    "  - dataset_summary.csv"
)

print()
print("Important:")
print(
    "  - Chronological train/test splitting was used."
)

print(
    "  - No random shuffling was performed."
)

print(
    "  - Future targets were excluded from predictors."
)

print(
    "  - Training samples whose future targets crossed "
    "the test boundary were removed."
)

print(
    "  - The 3276.7 °C ambient-temperature sentinel was corrected."
)

print(
    "  - Ambient temperature was excluded because no valid "
    "observations remain."
)

print(
    "  - No synthetic ambient temperature was created."
)

print(
    "  - No suspicious sensor measurements were deleted."
)

print(
    "  - No weather data was added."
)

print(
    "  - No interpolation was performed."
)

print()
print(
    "The leakage-controlled datasets are now ready "
    "for baseline model development."
)

print(
    "=" * 80
)