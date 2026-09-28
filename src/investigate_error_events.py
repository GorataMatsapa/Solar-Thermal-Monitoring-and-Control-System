from pathlib import Path
import pandas as pd


# ============================================================
# STMaCS — Investigate Prediction Error Events
# ============================================================

print("=" * 80)
print("STMaCS ERROR-EVENT TIME-SERIES INVESTIGATION")
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

# ------------------------------------------------------------
# Identify the worst error events
# ------------------------------------------------------------

worst = (
    pred.sort_values("absolute_error", ascending=False)
    .head(5)
)

# ------------------------------------------------------------
# Variables we want to inspect
# ------------------------------------------------------------

columns_to_inspect = [
    "timestamp",
    "buffer_top",
    "buffer_centre",
    "buffer_bottom",
    "collector_temp",
    "solar_radiation",
    "solar_flow",
    "solar_power",
    "hotwater_flow",
    "hotwater_power",
    "pwm",
    "solar_pump",
    "electric_heater",
]

available_columns = [
    col for col in columns_to_inspect
    if col in features.columns
]

# ------------------------------------------------------------
# Examine each event
# ------------------------------------------------------------

output_dir = (
    Path("data")
    / "processed"
    / "ml"
    / "baseline_results"
    / "error_events"
)

output_dir.mkdir(parents=True, exist_ok=True)

for i, row in worst.iterrows():

    event_time = row["timestamp"]

    print("\n" + "=" * 80)
    print("ERROR EVENT")
    print("=" * 80)

    print(f"Prediction timestamp: {event_time}")
    print(f"Actual future temperature:    {row['actual']:.2f} °C")
    print(f"Predicted future temperature: {row['predicted']:.2f} °C")
    print(f"Prediction error:              {row['error']:.2f} °C")

    # --------------------------------------------------------
    # 90 minutes before to 90 minutes after
    # --------------------------------------------------------

    start_time = event_time - pd.Timedelta(minutes=90)
    end_time = event_time + pd.Timedelta(minutes=90)

    event_data = features[
        (features["timestamp"] >= start_time)
        & (features["timestamp"] <= end_time)
    ][available_columns].copy()

    print("\nTime-series around event:")
    print(event_data.to_string(index=False))

    # --------------------------------------------------------
    # Save event
    # --------------------------------------------------------

    filename = (
        output_dir
        / f"event_{event_time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

    event_data.to_csv(filename, index=False)

    print(f"\nSaved: {filename}")

print("\n" + "=" * 80)
print("EVENT INVESTIGATION COMPLETE")
print("=" * 80)