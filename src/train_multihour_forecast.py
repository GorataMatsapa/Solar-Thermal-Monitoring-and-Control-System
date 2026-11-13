"""
STMaCS SYSTEM 455 - MULTI-HOUR FORECAST (direct models, 1 to 12 hours ahead)

Question answered
-----------------
How far ahead can the tank-top temperature be forecast, and do expected
demand and sunshine information help at longer horizons?

Method
------
For each horizon H (hours), one HistGradientBoosting model predicts the CHANGE
in buffer_top between now and now+H. Targets are built by looking up the actual
buffer_top at (timestamp + H) inside the same file, so train and test never mix.

Feature sets compared at every horizon
--------------------------------------
  CORE                   : the 54 portable features (tank/collector temps,
                           pump flow/state, time features, lags, deltas)
  CORE+DEMAND            : CORE + expected hot-water flow over the next H hours,
                           from a weekday x hour profile built on TRAIN data only
  CORE+DEMAND+SUN(oracle): the above + the MEASURED mean radiation over the
                           next H hours. This is a PERFECT-forecast upper bound,
                           not a realistic result. It shows the best case for
                           weather data. Later we replace it with API forecasts
                           (with noise) for a realistic number.

Baselines
---------
  Persistence : buffer_top stays as it is
  Hour-of-day : persistence + the typical train-set change for that decision hour

Usable-temperature check
------------------------
For each model we also report how often it correctly warns that the tank top
will be below USABLE_TEMP at the target time (the controller's real question):
  miss rate        = tank WAS below threshold but model did not warn (unsafe)
  false-alarm rate = tank was fine but model warned (wasted heating)
at margin 0 and at +SAFETY_MARGIN degrees.

Usage
-----
  python train_multihour_forecast.py            full run
  python train_multihour_forecast.py --quick    fast smoke test
  python train_multihour_forecast.py --ml-dir PATH   override data folder
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="fewer horizons and trees, for a fast smoke test")
parser.add_argument("--ml-dir", type=str, default=None,
                    help="folder containing system455_60min_train/test.csv")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = Path(args.ml_dir) if args.ml_dir else BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = ML_DIR / "baseline_results" / "multihour"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_H = [1, 4] if args.quick else [1, 2, 4, 8, 12]
MAX_ITER = 40 if args.quick else 300

USABLE_TEMP = 45.0        # degrees C at buffer_top considered "usable"
SAFETY_MARGIN = 2.0       # degrees C extra margin for the warning rule
EVENING_HOURS = (18, 21)  # "8pm-style" decision hours, inclusive
TOLERANCE_MIN = 3         # max gap when looking up the target timestamp
MIN_WINDOW_COVERAGE = 0.8 # min fraction of rows present in a future window
DEMAND_STEP_MIN = 15
RANDOM_STATE = 42

# Columns removed from every feature set (redundant)
REDUNDANT = ["pwm", "hour", "day_of_week", "day_of_year"]

# Additionally removed to form CORE
CORE_EXTRA_DROP = [
    "solar_power", "solar_temp_hot", "solar_temp_cold", "solar_loop_delta_t",
    "hotwater_flow", "hotwater_power", "hotwater_temp_hot",
    "hotwater_temp_cold", "hotwater_delta_t",
    "hotwater_flow_lag_1", "hotwater_flow_lag_3",
    "hotwater_flow_lag_6", "hotwater_flow_lag_12",
    "electric_heater",
    "solar_radiation", "solar_radiation_lag_1", "solar_radiation_lag_3",
    "solar_radiation_lag_6", "solar_radiation_lag_12",
]

NS_PER_MIN = 60 * 10**9
NS_PER_HOUR = 3600 * 10**9


# =============================================================================
# HELPERS
# =============================================================================

def lookup_nearest(ts_ns, values, query_ns, tol_ns):
    """Value at the row whose timestamp is nearest to each query time.
    NaN when no row lies within tol_ns."""
    idx = np.searchsorted(ts_ns, query_ns)
    left = np.clip(idx - 1, 0, len(ts_ns) - 1)
    right = np.clip(idx, 0, len(ts_ns) - 1)
    d_left = np.abs(query_ns - ts_ns[left])
    d_right = np.abs(ts_ns[right] - query_ns)
    use_right = d_right < d_left
    best = np.where(use_right, right, left)
    d_best = np.where(use_right, d_right, d_left)
    out = values[best].astype(float)
    out[d_best > tol_ns] = np.nan
    return out


def future_window_mean(ts_ns, values, start_ns, end_ns, min_rows):
    """Mean of values over rows with start < timestamp <= end.
    NaN when fewer than min_rows rows fall in the window."""
    cs = np.concatenate([[0.0], np.cumsum(values.astype(float))])
    i0 = np.searchsorted(ts_ns, start_ns, side="right")
    i1 = np.searchsorted(ts_ns, end_ns, side="right")
    n = i1 - i0
    out = np.full(len(start_ns), np.nan)
    ok = n >= max(min_rows, 1)
    out[ok] = (cs[i1[ok]] - cs[i0[ok]]) / n[ok]
    return out


def build_demand_profile(train_df):
    """Mean hotwater_flow by (weekday, hour), from TRAIN data only."""
    ts = train_df["timestamp"]
    grouped = train_df.groupby(
        [ts.dt.dayofweek, ts.dt.hour])["hotwater_flow"].mean()
    overall = train_df["hotwater_flow"].mean()
    profile = np.full((7, 24), overall)
    for (dow, hr), val in grouped.items():
        profile[int(dow), int(hr)] = val
    return profile


def expected_flow_next(timestamps, profile, horizon_h):
    """Average of the demand profile over the next horizon_h hours."""
    ts64 = timestamps.values.astype("datetime64[ns]")
    total = np.zeros(len(ts64))
    n_steps = 0
    for k in range(0, horizon_h * 60, DEMAND_STEP_MIN):
        t = pd.DatetimeIndex(ts64 + np.timedelta64(k, "m"))
        total += profile[t.dayofweek.values, t.hour.values]
        n_steps += 1
    return total / max(n_steps, 1)


def add_horizon_columns(df, horizon_h, profile, step_min):
    """Return a copy of df with target and horizon-specific features added.
    Rows without a valid target or window are dropped."""
    df = df.copy()
    ts_ns = df["timestamp"].values.astype("datetime64[ns]").astype("int64")
    h_ns = horizon_h * NS_PER_HOUR
    tol_ns = TOLERANCE_MIN * NS_PER_MIN

    df["target_abs"] = lookup_nearest(
        ts_ns, df["buffer_top"].to_numpy(), ts_ns + h_ns, tol_ns)
    df["target_delta"] = df["target_abs"] - df["buffer_top"]

    df["expected_flow_next"] = expected_flow_next(
        df["timestamp"], profile, horizon_h)

    min_rows = int(MIN_WINDOW_COVERAGE * horizon_h * 60 / step_min)
    df["future_radiation_mean"] = future_window_mean(
        ts_ns, df["solar_radiation"].to_numpy(), ts_ns, ts_ns + h_ns, min_rows)

    df = df.dropna(subset=["target_abs", "future_radiation_mean"])
    return df.reset_index(drop=True)


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def threshold_stats(y_true, y_pred, margin):
    """Miss rate and false-alarm rate for the 'will be unusable' warning."""
    actual_low = y_true < USABLE_TEMP
    warn = y_pred < (USABLE_TEMP + margin)
    n_low = int(actual_low.sum())
    n_ok = int((~actual_low).sum())
    miss = float((actual_low & ~warn).sum() / n_low) if n_low else np.nan
    false_alarm = float((~actual_low & warn).sum() / n_ok) if n_ok else np.nan
    return n_low, miss, false_alarm


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("MULTI-HOUR FORECAST" + ("  [QUICK MODE]" if args.quick else ""))
print("=" * 80)

train_file = ML_DIR / "system455_60min_train.csv"
test_file = ML_DIR / "system455_60min_test.csv"
print(f"Train: {train_file}")
print(f"Test : {test_file}")

train_raw = pd.read_csv(train_file)
test_raw = pd.read_csv(test_file)
train_raw["timestamp"] = pd.to_datetime(train_raw["timestamp"])
test_raw["timestamp"] = pd.to_datetime(test_raw["timestamp"])

assert train_raw["timestamp"].is_monotonic_increasing, "train not chronological"
assert test_raw["timestamp"].is_monotonic_increasing, "test not chronological"
assert train_raw["timestamp"].max() < test_raw["timestamp"].min(), "overlap"

all_predictors = [c for c in train_raw.columns
                  if c != "timestamp" and not c.startswith("target_")]
for name in REDUNDANT + CORE_EXTRA_DROP:
    if name not in all_predictors:
        raise ValueError(f"Column to drop not found in data: {name}")

FULL = [c for c in all_predictors if c not in REDUNDANT]
CORE = [c for c in FULL if c not in CORE_EXTRA_DROP]
print(f"\nCORE feature count: {len(CORE)} (expected 54)")

step_min = float(np.median(
    np.diff(train_raw["timestamp"].values.astype("datetime64[s]")
            .astype("int64"))) / 60.0)
print(f"Median sampling step: {step_min:.2f} minutes")

demand_profile = build_demand_profile(train_raw)  # TRAIN data only
print("Demand profile built from training data only.")

FEATURE_SETS = {
    "CORE": CORE,
    "CORE+DEMAND": CORE + ["expected_flow_next"],
    "CORE+DEMAND+SUN(oracle)": CORE + ["expected_flow_next",
                                       "future_radiation_mean"],
}

# =============================================================================
# MAIN LOOP
# =============================================================================

summary_rows = []
t_start = time.time()

for H in HORIZONS_H:
    print("\n" + "=" * 80)
    print(f"HORIZON: {H} hour(s) ahead")
    print("=" * 80)

    train_h = add_horizon_columns(train_raw, H, demand_profile, step_min)
    test_h = add_horizon_columns(test_raw, H, demand_profile, step_min)
    print(f"Train rows with valid target: {len(train_h):,}")
    print(f"Test rows with valid target : {len(test_h):,}")

    y_true = test_h["target_abs"].to_numpy()
    cur = test_h["buffer_top"].to_numpy()
    dec_hour = test_h["timestamp"].dt.hour.to_numpy()
    evening = (dec_hour >= EVENING_HOURS[0]) & (dec_hour <= EVENING_HOURS[1])

    predictions = {}

    # --- baselines ---
    predictions["Persistence"] = cur.copy()

    train_hour = train_h["timestamp"].dt.hour
    hour_delta = train_h.groupby(train_hour)["target_delta"].median()
    delta_by_hour = np.array([hour_delta.get(h, 0.0) for h in range(24)])
    predictions["Hour-of-day"] = cur + delta_by_hour[dec_hour]

    # --- models ---
    for set_name, feats in FEATURE_SETS.items():
        t0 = time.time()
        model = make_model()
        model.fit(train_h[feats], train_h["target_delta"])
        predictions[set_name] = cur + model.predict(test_h[feats])
        print(f"  trained {set_name:<26} ({len(feats)} features, "
              f"{time.time() - t0:.0f}s)")

    # --- metrics ---
    p_mae = mae(y_true, predictions["Persistence"])
    print(f"\n{'Model':<26}{'MAE':>8}{'gain%':>8}{'RMSE':>8}"
          f"{'eveMAE':>8}{'miss0%':>8}{'miss+m%':>9}{'fa+m%':>8}")

    for name, pred in predictions.items():
        m = mae(y_true, pred)
        r = rmse(y_true, pred)
        e_mae = mae(y_true[evening], pred[evening]) if evening.any() else np.nan
        n_low, miss0, fa0 = threshold_stats(y_true, pred, 0.0)
        _, miss_m, fa_m = threshold_stats(y_true, pred, SAFETY_MARGIN)
        gain = (p_mae - m) / p_mae * 100 if p_mae else np.nan

        print(f"{name:<26}{m:>8.3f}{gain:>8.1f}{r:>8.3f}{e_mae:>8.3f}"
              f"{miss0 * 100:>8.1f}{miss_m * 100:>9.1f}{fa_m * 100:>8.1f}")

        summary_rows.append({
            "horizon_h": H, "model": name,
            "mae": m, "rmse": r, "mae_gain_vs_persistence_pct": gain,
            "evening_mae": e_mae,
            "bias": float(np.mean(y_true - pred)),
            "p95_abs_error": float(np.percentile(np.abs(y_true - pred), 95)),
            "n_test": len(y_true),
            "n_below_threshold": n_low,
            "miss_rate_margin0": miss0, "false_alarm_margin0": fa0,
            "miss_rate_with_margin": miss_m, "false_alarm_with_margin": fa_m,
        })

    out = pd.DataFrame({"timestamp": test_h["timestamp"], "actual": y_true})
    for name, pred in predictions.items():
        out[name] = pred
    out.to_csv(OUT_DIR / f"predictions_{H}h.csv", index=False)

# =============================================================================
# SAVE + SUMMARY
# =============================================================================

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "multihour_summary.csv", index=False)

print("\n" + "=" * 80)
print("MAE (degrees C) BY HORIZON")
print("=" * 80)
pivot = summary.pivot(index="model", columns="horizon_h", values="mae")
print(pivot.round(3).to_string())

print("\nThe test window has "
      f"{int(summary['n_below_threshold'].iloc[0]):,} rows below "
      f"{USABLE_TEMP:.0f} C at the 1h target - if the miss/false-alarm "
      "columns above rest on very few rows, treat them cautiously.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in pivot.index:
        ax.plot(pivot.columns, pivot.loc[name], marker="o", label=name)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE of buffer_top (C)")
    ax.set_title("Forecast error vs horizon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_vs_horizon.png", dpi=150)
    print("Saved plot: error_vs_horizon.png")
except Exception:
    print("(matplotlib not available - skipped plot)")

print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
print(f"Saved to: {OUT_DIR}")
