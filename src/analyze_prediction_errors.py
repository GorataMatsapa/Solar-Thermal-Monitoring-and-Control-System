from pathlib import Path
import pandas as pd


# ============================================================
# STMaCS — Prediction Error Diagnostic
# ============================================================

print("=" * 80)
print("STMaCS PREDICTION ERROR DIAGNOSTIC")
print("=" * 80)

# ------------------------------------------------------------
# 1. Load prediction file
# ------------------------------------------------------------

prediction_file = (
    Path("data")
    / "processed"
    / "ml"
    / "baseline_results"
    / "predictions"
    / "system455_60min_histgradientboosting_predictions.csv"
)

df = pd.read_csv(prediction_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nPrediction file:")
print(prediction_file)

print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# ------------------------------------------------------------
# 2. Basic error statistics
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR STATISTICS")
print("=" * 80)

print(f"MAE:                {df['absolute_error'].mean():.4f} °C")
print(f"RMSE:               {(df['error'] ** 2).mean() ** 0.5:.4f} °C")
print(f"Mean error:         {df['error'].mean():.4f} °C")
print(f"Maximum abs error:  {df['absolute_error'].max():.4f} °C")

# ------------------------------------------------------------
# 3. Error distribution
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR DISTRIBUTION")
print("=" * 80)

thresholds = [0.25, 0.5, 1, 2, 5, 10]

for threshold in thresholds:
    count = (df["absolute_error"] > threshold).sum()
    percentage = count / len(df) * 100

    print(
        f"|error| > {threshold:>5.2f} °C : "
        f"{count:>6,} predictions "
        f"({percentage:>6.2f}%)"
    )

# ------------------------------------------------------------
# 4. Worst 20 predictions
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("20 LARGEST PREDICTION ERRORS")
print("=" * 80)

worst = (
    df.sort_values("absolute_error", ascending=False)
    .head(20)
    .copy()
)

print(
    worst[
        [
            "timestamp",
            "actual",
            "predicted",
            "error",
            "absolute_error",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 5. Largest underpredictions
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("10 LARGEST UNDERPREDICTIONS")
print("=" * 80)

under = (
    df.sort_values("error", ascending=False)
    .head(10)
)

print(
    under[
        [
            "timestamp",
            "actual",
            "predicted",
            "error",
            "absolute_error",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 6. Largest overpredictions
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("10 LARGEST OVERPREDICTIONS")
print("=" * 80)

over = (
    df.sort_values("error", ascending=True)
    .head(10)
)

print(
    over[
        [
            "timestamp",
            "actual",
            "predicted",
            "error",
            "absolute_error",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 7. Temperature range
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ACTUAL / PREDICTED TEMPERATURE RANGE")
print("=" * 80)

print(
    f"Actual temperature range:    "
    f"{df['actual'].min():.2f} → {df['actual'].max():.2f} °C"
)

print(
    f"Predicted temperature range: "
    f"{df['predicted'].min():.2f} → {df['predicted'].max():.2f} °C"
)

print("\nDiagnostic complete.")