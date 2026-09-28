from pathlib import Path
import pandas as pd


# ============================================================
# STMaCS — Investigate Worst Prediction Errors
# ============================================================

print("=" * 80)
print("STMaCS WORST-ERROR SYSTEM-STATE INVESTIGATION")
print("=" * 80)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

prediction_file = (
    Path("data")
    / "processed"
    / "ml"
    / "baseline_results"
    / "predictions"
    / "system455_60min_histgradientboosting_predictions.csv"
)

feature_file = (
    Path("data")
    / "processed"
    / "system455_features.csv"
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

pred = pd.read_csv(prediction_file)
features = pd.read_csv(feature_file)

pred["timestamp"] = pd.to_datetime(pred["timestamp"])
features["timestamp"] = pd.to_datetime(features["timestamp"])

print(f"\nPrediction rows: {len(pred):,}")
print(f"Feature rows:    {len(features):,}")

# ------------------------------------------------------------
# Identify useful columns
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE COLUMNS")
print("=" * 80)

for col in features.columns:
    print(col)

# ------------------------------------------------------------
# Merge prediction errors with system state
# ------------------------------------------------------------

merged = pred.merge(
    features,
    on="timestamp",
    how="left",
    suffixes=("", "_feature")
)

print("\nMerged rows:", len(merged))

# ------------------------------------------------------------
# Select worst 20 errors
# ------------------------------------------------------------

worst = (
    merged
    .sort_values("absolute_error", ascending=False)
    .head(20)
)

# ------------------------------------------------------------
# Automatically identify useful thermal/operational columns
# ------------------------------------------------------------

keywords = [
    "buffer",
    "collector",
    "solar",
    "flow",
    "power",
    "heater",
    "pump",
    "radiation",
    "pwm",
    "temperature",
    "temp",
]

candidate_columns = []

for col in features.columns:
    col_lower = col.lower()

    if any(keyword in col_lower for keyword in keywords):
        candidate_columns.append(col)

# Remove timestamp and future target columns
candidate_columns = [
    col for col in candidate_columns
    if col != "timestamp"
    and "future" not in col.lower()
    and "target" not in col.lower()
    and "next" not in col.lower()
]

# Avoid an enormous output
candidate_columns = candidate_columns[:40]

output_columns = [
    "timestamp",
    "actual",
    "predicted",
    "error",
    "absolute_error",
]

output_columns += [
    col for col in candidate_columns
    if col in worst.columns
]

# Remove duplicate column names while preserving order
output_columns = list(dict.fromkeys(output_columns))

# ------------------------------------------------------------
# Display worst cases with system state
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("WORST 20 ERRORS + SYSTEM STATE")
print("=" * 80)

print(
    worst[output_columns]
    .to_string(index=False)
)

# ------------------------------------------------------------
# Save investigation file
# ------------------------------------------------------------

output_file = (
    Path("data")
    / "processed"
    / "ml"
    / "baseline_results"
    / "worst_error_investigation.csv"
)

worst[output_columns].to_csv(output_file, index=False)

print("\nSaved investigation file:")
print(output_file)

print("\nInvestigation complete.")