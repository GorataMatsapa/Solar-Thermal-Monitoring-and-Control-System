"""
STMaCS SYSTEM 455 - CLEANED-UP ABLATION RUN (60-minute horizon)

What this does
--------------
Trains HistGradientBoosting twice on the same data, with two feature sets,
and compares both against persistence:

  FULL : everything useful (heat meter, draw-off, heater state, radiation),
         minus only redundant columns (pwm, raw hour/day_of_week/day_of_year)
  CORE : only what a minimal installation provides - tank temperatures,
         collector temperature, pump flow/state, time features, and their
         lags/deltas

Both models predict the CHANGE in buffer_top (target - current buffer_top),
then add the current value back, so the metrics are in real degrees C.

The gap between FULL and CORE shows how much the heat meter / draw-off /
heater / radiation inputs add.

Place this file in the same folder as your other scripts
(it uses the same BASE_DIR logic as train_baseline_models.py).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = ML_DIR / "baseline_results" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 60
TARGET = f"target_buffer_top_{HORIZON}min"
RANDOM_STATE = 42

# Redundant columns removed from BOTH feature sets
REDUNDANT = ["pwm", "hour", "day_of_week", "day_of_year"]

# Additionally removed from the CORE set only
HEAT_METER = [
    "solar_power", "solar_temp_hot", "solar_temp_cold", "solar_loop_delta_t",
]
DRAW_OFF = [
    "hotwater_flow", "hotwater_power", "hotwater_temp_hot",
    "hotwater_temp_cold", "hotwater_delta_t",
    "hotwater_flow_lag_1", "hotwater_flow_lag_3",
    "hotwater_flow_lag_6", "hotwater_flow_lag_12",
]
HEATER = ["electric_heater"]
RADIATION = [
    "solar_radiation", "solar_radiation_lag_1", "solar_radiation_lag_3",
    "solar_radiation_lag_6", "solar_radiation_lag_12",
]

CORE_EXTRA_DROP = HEAT_METER + DRAW_OFF + HEATER + RADIATION


# =============================================================================
# HELPERS
# =============================================================================

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def pct_gain(base, new):
    return (base - new) / base * 100 if base != 0 else np.nan


# =============================================================================
# LOAD DATA
# =============================================================================

train_file = ML_DIR / f"system455_{HORIZON}min_train.csv"
test_file = ML_DIR / f"system455_{HORIZON}min_test.csv"

print("=" * 80)
print(f"ABLATION RUN - {HORIZON}-MINUTE HORIZON")
print("=" * 80)
print(f"Train: {train_file}")
print(f"Test : {test_file}")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

assert train_df["timestamp"].is_monotonic_increasing, "train not chronological"
assert test_df["timestamp"].is_monotonic_increasing, "test not chronological"
assert train_df["timestamp"].max() < test_df["timestamp"].min(), "overlap"

print(f"\nTraining rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")

# All predictors = everything except timestamp and any target column
all_predictors = [
    c for c in train_df.columns
    if c != "timestamp" and not c.startswith("target_")
]

# Make sure every column we plan to drop actually exists (catch typos early)
for name in REDUNDANT + CORE_EXTRA_DROP:
    if name not in all_predictors:
        raise ValueError(f"Column to drop not found in data: {name}")

FULL_FEATURES = [c for c in all_predictors if c not in REDUNDANT]
CORE_FEATURES = [c for c in FULL_FEATURES if c not in CORE_EXTRA_DROP]

print(f"\nAll predictors in file : {len(all_predictors)}")
print(f"FULL feature set       : {len(FULL_FEATURES)}  (expected 73)")
print(f"CORE feature set       : {len(CORE_FEATURES)}  (expected 54)")

y_train_abs = train_df[TARGET]
y_test_abs = test_df[TARGET]

# Change target: how much buffer_top moves over the horizon
y_train_delta = y_train_abs - train_df["buffer_top"]

# =============================================================================
# PERSISTENCE BASELINE
# =============================================================================

persistence_pred = test_df["buffer_top"].to_numpy()
p_mae = mae(y_test_abs, persistence_pred)
p_rmse = rmse(y_test_abs, persistence_pred)

print("\n" + "-" * 80)
print("PERSISTENCE")
print("-" * 80)
print(f"MAE : {p_mae:.4f} C")
print(f"RMSE: {p_rmse:.4f} C")

# =============================================================================
# TRAIN + EVALUATE BOTH FEATURE SETS
# =============================================================================

feature_sets = {
    "FULL": FULL_FEATURES,
    "CORE": CORE_FEATURES,
}

results = []
predictions = {}

for name, features in feature_sets.items():
    print("\n" + "-" * 80)
    print(f"TRAINING HistGradientBoosting - {name} ({len(features)} features)")
    print("-" * 80)

    model = make_model()
    model.fit(train_df[features], y_train_delta)

    # Predict the change, then add current buffer_top back
    delta_pred = model.predict(test_df[features])
    pred = test_df["buffer_top"].to_numpy() + delta_pred
    predictions[name] = pred

    m_mae = mae(y_test_abs, pred)
    m_rmse = rmse(y_test_abs, pred)
    errors = y_test_abs.to_numpy() - pred

    print(f"MAE : {m_mae:.4f} C   ({pct_gain(p_mae, m_mae):+.2f}% vs persistence)")
    print(f"RMSE: {m_rmse:.4f} C   ({pct_gain(p_rmse, m_rmse):+.2f}% vs persistence)")
    print(f"Mean error        : {errors.mean():+.4f} C")
    print(f"95th pct abs error: {np.percentile(np.abs(errors), 95):.4f} C")
    print(f"Max abs error     : {np.abs(errors).max():.4f} C")

    results.append({
        "feature_set": name,
        "n_features": len(features),
        "mae": m_mae,
        "rmse": m_rmse,
        "mae_gain_vs_persistence_pct": pct_gain(p_mae, m_mae),
        "rmse_gain_vs_persistence_pct": pct_gain(p_rmse, m_rmse),
        "mean_error": errors.mean(),
        "p95_abs_error": np.percentile(np.abs(errors), 95),
        "max_abs_error": np.abs(errors).max(),
    })

results.append({
    "feature_set": "Persistence",
    "n_features": 1,
    "mae": p_mae,
    "rmse": p_rmse,
    "mae_gain_vs_persistence_pct": 0.0,
    "rmse_gain_vs_persistence_pct": 0.0,
    "mean_error": (y_test_abs.to_numpy() - persistence_pred).mean(),
    "p95_abs_error": np.percentile(
        np.abs(y_test_abs.to_numpy() - persistence_pred), 95),
    "max_abs_error": np.abs(y_test_abs.to_numpy() - persistence_pred).max(),
})

# =============================================================================
# WHERE DOES THE GAIN COME FROM? (breakdown by operating condition)
# =============================================================================

print("\n" + "=" * 80)
print("MAE BY OPERATING CONDITION (test set)")
print("=" * 80)

conditions = {}
if "hotwater_flow" in test_df.columns:
    conditions["Draw-off active (hotwater_flow > 0)"] = (
        test_df["hotwater_flow"] > 0).to_numpy()
    conditions["No draw-off (hotwater_flow == 0)"] = (
        test_df["hotwater_flow"] == 0).to_numpy()
if "solar_pump" in test_df.columns:
    conditions["Solar pump ON"] = (test_df["solar_pump"] == 1).to_numpy()
    conditions["Solar pump OFF"] = (test_df["solar_pump"] == 0).to_numpy()
if "electric_heater" in test_df.columns:
    conditions["Heater ON"] = (test_df["electric_heater"] == 1).to_numpy()
conditions["All rows"] = np.ones(len(test_df), dtype=bool)

y_true_arr = y_test_abs.to_numpy()
breakdown_rows = []

print(f"\n{'Condition':<38}{'n':>8}{'Persist':>10}{'FULL':>10}{'CORE':>10}")
for cond_name, mask in conditions.items():
    n = int(mask.sum())
    if n == 0:
        continue
    row = {
        "condition": cond_name,
        "n_rows": n,
        "persistence_mae": mae(y_true_arr[mask], persistence_pred[mask]),
        "full_mae": mae(y_true_arr[mask], predictions["FULL"][mask]),
        "core_mae": mae(y_true_arr[mask], predictions["CORE"][mask]),
    }
    breakdown_rows.append(row)
    print(f"{cond_name:<38}{n:>8,}"
          f"{row['persistence_mae']:>10.3f}"
          f"{row['full_mae']:>10.3f}"
          f"{row['core_mae']:>10.3f}")

# =============================================================================
# SAVE
# =============================================================================

pd.DataFrame(results).to_csv(OUT_DIR / "ablation_summary.csv", index=False)
pd.DataFrame(breakdown_rows).to_csv(
    OUT_DIR / "ablation_breakdown.csv", index=False)

pd.DataFrame({
    "timestamp": test_df["timestamp"],
    "actual": y_true_arr,
    "persistence": persistence_pred,
    "full": predictions["FULL"],
    "core": predictions["CORE"],
}).to_csv(OUT_DIR / "ablation_predictions.csv", index=False)

pd.Series(FULL_FEATURES, name="feature").to_csv(
    OUT_DIR / "full_feature_list.csv", index=False)
pd.Series(CORE_FEATURES, name="feature").to_csv(
    OUT_DIR / "core_feature_list.csv", index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved to: {OUT_DIR}")
print("  - ablation_summary.csv")
print("  - ablation_breakdown.csv")
print("  - ablation_predictions.csv")
print("  - full_feature_list.csv / core_feature_list.csv")
