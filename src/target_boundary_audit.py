"""
STMaCS SYSTEM 455
ACTUAL TARGET TIMESTAMP / TRAIN-TEST BOUNDARY AUDIT

Purpose
-------
Reproduce the exact target construction used in feature_engineering.py
and determine whether any training samples have actual future targets
whose matched measurement occurs inside the test period.

This script DOES NOT modify any project files.
It DOES NOT retrain any ML models.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "system455_clean.csv"
)

TRAIN_FRACTION = 0.80

HORIZONS = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}

TOLERANCE_MINUTES = 2


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STMaCS SYSTEM 455 ACTUAL TARGET TIMESTAMP AUDIT")
print("=" * 80)

print()
print("Purpose:")
print("  Reproduce the exact merge_asof target matching used in")
print("  feature_engineering.py and inspect the actual matched")
print("  future timestamps relative to the train/test boundary.")

print()
print("No files will be modified.")
print("No ML models will be trained.")


# =============================================================================
# LOAD DATA
# =============================================================================

print()
print("=" * 80)
print("1. LOADING SYSTEM 455 DATA")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="raise"
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(f"\nRows: {len(df):,}")
print(f"First timestamp: {df['timestamp'].iloc[0]}")
print(f"Last timestamp:  {df['timestamp'].iloc[-1]}")


# =============================================================================
# CREATE EXACT TRAIN / TEST BOUNDARY
# =============================================================================

print()
print("=" * 80)
print("2. TRAIN / TEST BOUNDARY")
print("=" * 80)

total_rows = len(df)

train_rows = int(
    total_rows * TRAIN_FRACTION
)

train_df = df.iloc[:train_rows].copy()
test_df = df.iloc[train_rows:].copy()

train_end = train_df["timestamp"].iloc[-1]
test_start = test_df["timestamp"].iloc[0]

print(f"\nTotal rows:     {total_rows:,}")
print(f"Training rows:  {len(train_df):,}")
print(f"Testing rows:   {len(test_df):,}")

print()
print(f"Last training timestamp:")
print(f"  {train_end}")

print()
print(f"First testing timestamp:")
print(f"  {test_start}")

print()
print(
    "Boundary separation: "
    f"{(test_start - train_end).total_seconds() / 60:.2f} minutes"
)


# =============================================================================
# REPRODUCE EXACT TARGET MATCHING
# =============================================================================

print()
print("=" * 80)
print("3. REPRODUCING ACTUAL merge_asof TARGET MATCHING")
print("=" * 80)

# This is the same target reference used by feature_engineering.py.

target_reference = df[
    ["timestamp", "buffer_top"]
].copy()

target_reference = target_reference.rename(
    columns={
        "buffer_top": "future_value"
    }
)

target_reference = (
    target_reference
    .set_index("timestamp")
)

source = (
    target_reference
    .reset_index()
    .sort_values("timestamp")
)

all_audits = []


for label, minutes in HORIZONS.items():

    print()
    print("-" * 80)
    print(f"{minutes}-MINUTE TARGET")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Exact future-time construction from feature_engineering.py
    # -------------------------------------------------------------------------

    future_times = (
        df["timestamp"]
        + pd.Timedelta(minutes=minutes)
    )

    lookup = pd.DataFrame(
        {
            "future_time": future_times,
            "row_id": np.arange(len(df)),
        }
    )

    lookup = (
        lookup
        .sort_values("future_time")
    )

    # -------------------------------------------------------------------------
    # Exact merge_asof operation
    # -------------------------------------------------------------------------

    matched = pd.merge_asof(
        lookup,
        source,
        left_on="future_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            minutes=TOLERANCE_MINUTES
        ),
    )

    matched = (
        matched
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    # Rename actual matched timestamp for clarity.
    matched = matched.rename(
        columns={
            "timestamp": "actual_target_timestamp"
        }
    )

    # -------------------------------------------------------------------------
    # Attach original observation timestamp
    # -------------------------------------------------------------------------

    matched["observation_timestamp"] = df["timestamp"].to_numpy()

    matched["future_time"] = (
        matched["observation_timestamp"]
        + pd.Timedelta(minutes=minutes)
    )

    # -------------------------------------------------------------------------
    # Calculate matching error
    # -------------------------------------------------------------------------

    matched["target_offset_minutes"] = (
        (
            matched["actual_target_timestamp"]
            - matched["future_time"]
        )
        .dt.total_seconds()
        / 60
    )

    # -------------------------------------------------------------------------
    # Determine train/test membership
    # -------------------------------------------------------------------------

    matched["observation_is_train"] = (
        matched["observation_timestamp"] < test_start
    )

    matched["target_is_test"] = (
        matched["actual_target_timestamp"] >= test_start
    )

    # Only training observations whose ACTUAL matched target
    # is inside the test period.
    boundary_crossing = matched[
        matched["observation_is_train"]
        & matched["target_is_test"]
    ].copy()

    # Only training observations with a successfully matched target.
    train_matched = matched[
        matched["observation_is_train"]
        & matched["actual_target_timestamp"].notna()
    ]

    # -------------------------------------------------------------------------
    # Basic statistics
    # -------------------------------------------------------------------------

    valid_targets = matched[
        "actual_target_timestamp"
    ].notna().sum()

    missing_targets = matched[
        "actual_target_timestamp"
    ].isna().sum()

    print()
    print(f"Valid target matches:   {valid_targets:,}")
    print(f"Missing target matches: {missing_targets:,}")

    print()
    print("Target matching offset:")
    print(
        f"  Mean:   "
        f"{train_matched['target_offset_minutes'].mean():+.4f} min"
    )
    print(
        f"  Median: "
        f"{train_matched['target_offset_minutes'].median():+.4f} min"
    )
    print(
        f"  Minimum:"
        f" {train_matched['target_offset_minutes'].min():+.4f} min"
    )
    print(
        f"  Maximum:"
        f" {train_matched['target_offset_minutes'].max():+.4f} min"
    )

    # -------------------------------------------------------------------------
    # Boundary result
    # -------------------------------------------------------------------------

    print()
    print("ACTUAL TRAINING TARGETS CROSSING INTO TEST PERIOD:")
    print(
        f"  {len(boundary_crossing):,}"
    )

    if len(train_matched) > 0:

        percentage = (
            len(boundary_crossing)
            / len(train_matched)
            * 100
        )

        print(
            f"  Percentage of matched training samples: "
            f"{percentage:.6f}%"
        )

    # -------------------------------------------------------------------------
    # Print affected observations
    # -------------------------------------------------------------------------

    if len(boundary_crossing) > 0:

        print()
        print("Affected samples:")

        display_columns = [
            "observation_timestamp",
            "future_time",
            "actual_target_timestamp",
            "target_offset_minutes",
        ]

        print(
            boundary_crossing[
                display_columns
            ].to_string(index=False)
        )

    else:

        print()
        print("[NONE] No actual training targets cross the test boundary.")

    # -------------------------------------------------------------------------
    # Calculate safe training rows
    # -------------------------------------------------------------------------

    train_with_target = matched[
        matched["observation_is_train"]
        & matched["actual_target_timestamp"].notna()
    ].copy()

    safe_train = train_with_target[
        train_with_target["actual_target_timestamp"] < test_start
    ]

    print()
    print("TRAINING SAMPLE COUNTS:")
    print(
        f"  Training observations with valid targets: "
        f"{len(train_with_target):,}"
    )

    print(
        f"  Safe training observations: "
        f"{len(safe_train):,}"
    )

    print(
        f"  Boundary-crossing observations: "
        f"{len(boundary_crossing):,}"
    )

    # -------------------------------------------------------------------------
    # Store audit information
    # -------------------------------------------------------------------------

    all_audits.append({
        "horizon_min": minutes,
        "valid_targets": valid_targets,
        "missing_targets": missing_targets,
        "training_targets_checked": len(train_with_target),
        "boundary_crossing": len(boundary_crossing),
        "boundary_crossing_pct": (
            len(boundary_crossing)
            / len(train_with_target)
            * 100
            if len(train_with_target) > 0
            else 0
        ),
        "safe_training_rows": len(safe_train),
        "mean_target_offset_min": (
            train_matched["target_offset_minutes"].mean()
        ),
        "median_target_offset_min": (
            train_matched["target_offset_minutes"].median()
        ),
        "max_abs_target_offset_min": (
            train_matched["target_offset_minutes"]
            .abs()
            .max()
        ),
    })


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("4. FINAL ACTUAL-TARGET BOUNDARY SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(all_audits)

print()

print(
    summary_df[
        [
            "horizon_min",
            "training_targets_checked",
            "boundary_crossing",
            "boundary_crossing_pct",
            "safe_training_rows",
        ]
    ].to_string(index=False)
)


# =============================================================================
# TARGET MATCHING QUALITY SUMMARY
# =============================================================================

print()
print("=" * 80)
print("5. TARGET MATCHING QUALITY")
print("=" * 80)

print()

print(
    summary_df[
        [
            "horizon_min",
            "mean_target_offset_min",
            "median_target_offset_min",
            "max_abs_target_offset_min",
        ]
    ].to_string(index=False)
)


# =============================================================================
# CONCLUSION
# =============================================================================

print()
print("=" * 80)
print("AUDIT CONCLUSION")
print("=" * 80)

total_boundary = summary_df[
    "boundary_crossing"
].sum()

if total_boundary == 0:

    print()
    print(
        "[PASS] No actual training targets were found inside "
        "the test period."
    )

else:

    print()
    print(
        f"[FOUND] {total_boundary:,} actual training samples "
        "have targets inside the test period."
    )

    print()
    print(
        "These samples should be excluded from a final strictly "
        "isolated chronological training set."
    )

print()
print("No files were modified.")
print("No models were trained.")
print()
print("=" * 80)
print("END OF TARGET TIMESTAMP AUDIT")
print("=" * 80)