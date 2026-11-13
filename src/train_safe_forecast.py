"""
STMaCS SYSTEM 455 - SAFE (CONSERVATIVE) MULTI-HOUR FORECAST

Why this script exists
----------------------
The previous run showed the average forecast is good, but it missed too many
"tank will be below the usable temperature" events at long horizons (a miss
means cold water). This script makes the forecast safe in three ways:

  1. ANCHORING: models predict the residual on top of the hour-of-day
     baseline (persistence + typical train-set change for that decision hour),
     so they cannot do worse than that baseline in quiet periods.
  2. QUANTILE MODELS: instead of only predicting the average, we also train
     models that predict a pessimistic value (20th, 10th, 5th percentile).
     "The tank will be at least X degrees with 90% confidence" is the right
     kind of forecast for a safety-critical controller.
  3. MARGIN SWEEP: for every predictor we sweep a safety margin and report the
     trade-off between missed warnings (unsafe) and unnecessary warnings
     (wasted heating). The key result is: "to keep misses under 5%, you need
     margin M, which costs F% false alarms".

Models per horizon (feature set: CORE, the 54 portable features)
----------------------------------------------------------------
  Persistence, Hour-of-day      : baselines
  Mean (previous)               : the previous run's model, for reference
  Q50 / Q20 / Q10 / Q05 anchored: quantile models on the anchored residual

Definitions
-----------
  usable          : buffer_top >= USABLE_TEMP at the target time
  warning rule    : warn if  prediction < USABLE_TEMP + margin
  miss rate       : tank WAS below USABLE_TEMP but no warning was given
  false-alarm rate: tank was fine but a warning was given
  warn rate       : fraction of all decision times with a warning

Usage
-----
  python train_safe_forecast.py                 full run (~10 min)
  python train_safe_forecast.py --quick         fast smoke test
  python train_safe_forecast.py --with-sun      add measured future radiation
                                                (oracle upper bound)
  python train_safe_forecast.py --ml-dir PATH   override data folder
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
parser.add_argument("--quick", action="store_true")
parser.add_argument("--with-sun", action="store_true",
                    help="add measured future radiation (oracle upper bound)")
parser.add_argument("--ml-dir", type=str, default=None)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = Path(args.ml_dir) if args.ml_dir else BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = ML_DIR / "baseline_results" / "safe_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_H = [1, 4] if args.quick else [1, 2, 4, 8, 12]
MAX_ITER = 40 if args.quick else 300
QUANTILES = [0.5, 0.2, 0.1, 0.05]

USABLE_TEMP = 45.0
EVENING_HOURS = (18, 21)
MISS_TARGETS = [0.05, 0.10]               # acceptable miss rates to report
MARGINS = np.arange(-2.0, 15.01, 0.25)    # safety-margin sweep (degrees C)
TOLERANCE_MIN = 3
MIN_WINDOW_COVERAGE = 0.8
RANDOM_STATE = 42

REDUNDANT = ["pwm", "hour", "day_of_week", "day_of_year"]
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
    cs = np.concatenate([[0.0], np.cumsum(values.astype(float))])
    i0 = np.searchsorted(ts_ns, start_ns, side="right")
    i1 = np.searchsorted(ts_ns, end_ns, side="right")
    n = i1 - i0
    out = np.full(len(start_ns), np.nan)
    ok = n >= max(min_rows, 1)
    out[ok] = (cs[i1[ok]] - cs[i0[ok]]) / n[ok]
    return out


def add_horizon_columns(df, horizon_h, step_min, need_sun):
    df = df.copy()
    ts_ns = df["timestamp"].values.astype("datetime64[ns]").astype("int64")
    h_ns = horizon_h * NS_PER_HOUR
    tol_ns = TOLERANCE_MIN * NS_PER_MIN

    df["target_abs"] = lookup_nearest(
        ts_ns, df["buffer_top"].to_numpy(), ts_ns + h_ns, tol_ns)
    df["target_delta"] = df["target_abs"] - df["buffer_top"]
    subset = ["target_abs"]

    if need_sun:
        min_rows = int(MIN_WINDOW_COVERAGE * horizon_h * 60 / step_min)
        df["future_radiation_mean"] = future_window_mean(
            ts_ns, df["solar_radiation"].to_numpy(),
            ts_ns, ts_ns + h_ns, min_rows)
        subset.append("future_radiation_mean")

    return df.dropna(subset=subset).reset_index(drop=True)


def make_model(loss="squared_error", quantile=None):
    kwargs = dict(
        loss=loss,
        max_iter=MAX_ITER,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def operating_curve(y_true, pred, margins):
    """Miss, false-alarm and warn rates for each margin."""
    actual_low = y_true < USABLE_TEMP
    n_low = int(actual_low.sum())
    n_ok = int((~actual_low).sum())
    miss = np.full(len(margins), np.nan)
    fa = np.full(len(margins), np.nan)
    warn_rate = np.zeros(len(margins))
    for i, m in enumerate(margins):
        warn = pred < (USABLE_TEMP + m)
        warn_rate[i] = warn.mean()
        if n_low:
            miss[i] = (actual_low & ~warn).sum() / n_low
        if n_ok:
            fa[i] = (~actual_low & warn).sum() / n_ok
    return miss, fa, warn_rate, n_low


def operating_point(miss, fa, warn_rate, margins, target):
    """Smallest margin whose miss rate is <= target. None if unreachable."""
    ok = np.where(miss <= target)[0]
    if len(ok) == 0:
        return None
    i = ok[0]
    return margins[i], fa[i], warn_rate[i]


def fmt(op, which):
    if op is None:
        return "   n/a"
    return f"{op[which] * 100:6.1f}" if which > 0 else f"{op[0]:+6.2f}"


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("SAFE MULTI-HOUR FORECAST"
      + ("  [QUICK]" if args.quick else "")
      + ("  [+SUN oracle]" if args.with_sun else ""))
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
FEATURES = CORE + (["future_radiation_mean"] if args.with_sun else [])
print(f"\nFeature count: {len(FEATURES)} "
      f"(CORE {len(CORE)}{' + oracle radiation' if args.with_sun else ''})")

step_min = float(np.median(
    np.diff(train_raw["timestamp"].values.astype("datetime64[s]")
            .astype("int64"))) / 60.0)
print(f"Median sampling step: {step_min:.2f} minutes")
print(f"Usable threshold: {USABLE_TEMP:.0f} C at buffer_top")

# =============================================================================
# MAIN LOOP
# =============================================================================

summary_rows = []
curve_rows = []
t_start = time.time()

for H in HORIZONS_H:
    print("\n" + "=" * 80)
    print(f"HORIZON: {H} hour(s) ahead")
    print("=" * 80)

    train_h = add_horizon_columns(train_raw, H, step_min, args.with_sun)
    test_h = add_horizon_columns(test_raw, H, step_min, args.with_sun)
    print(f"Train rows: {len(train_h):,}   Test rows: {len(test_h):,}")

    y_true = test_h["target_abs"].to_numpy()
    cur = test_h["buffer_top"].to_numpy()
    cur_train = train_h["buffer_top"].to_numpy()
    dec_hour = test_h["timestamp"].dt.hour.to_numpy()
    dec_hour_train = train_h["timestamp"].dt.hour.to_numpy()
    evening = (dec_hour >= EVENING_HOURS[0]) & (dec_hour <= EVENING_HOURS[1])

    # Hour-of-day anchor, from TRAIN only
    hour_delta = train_h.groupby(dec_hour_train)["target_delta"].median()
    delta_by_hour = np.array([hour_delta.get(h, 0.0) for h in range(24)])
    anchor_train = cur_train + delta_by_hour[dec_hour_train]
    anchor_test = cur + delta_by_hour[dec_hour]
    resid_train = train_h["target_abs"].to_numpy() - anchor_train

    predictions = {
        "Persistence": cur.copy(),
        "Hour-of-day": anchor_test.copy(),
    }

    # Reference: previous run's mean model (unanchored delta target)
    t0 = time.time()
    ref = make_model()
    ref.fit(train_h[FEATURES], train_h["target_delta"])
    predictions["Mean (previous)"] = cur + ref.predict(test_h[FEATURES])
    print(f"  trained Mean (previous)   ({time.time() - t0:.0f}s)")

    # Anchored quantile models
    for q in QUANTILES:
        t0 = time.time()
        m = make_model(loss="quantile", quantile=q)
        m.fit(train_h[FEATURES], resid_train)
        predictions[f"Q{int(q * 100):02d} anchored"] = (
            anchor_test + m.predict(test_h[FEATURES]))
        print(f"  trained Q{int(q * 100):02d} anchored        "
              f"({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    # Evaluate: accuracy, calibration, and the safety trade-off
    # ------------------------------------------------------------------
    n_low_all = int((y_true < USABLE_TEMP).sum())
    n_low_eve = int((y_true[evening] < USABLE_TEMP).sum()) if evening.any() else 0
    print(f"\nTest rows below {USABLE_TEMP:.0f} C at target time: "
          f"{n_low_all:,} of {len(y_true):,} "
          f"(evening decisions: {n_low_eve:,} of {int(evening.sum()):,})")

    hdr = (f"\n{'Model':<18}{'MAE':>7}{'cover%':>8} | "
           f"{'miss<=5%: margin':>17}{'warn%':>7}{'FA%':>7} | "
           f"{'miss<=10%: margin':>18}{'warn%':>7}{'FA%':>7}")
    print(hdr)

    for name, pred in predictions.items():
        miss, fa, warn, n_low = operating_curve(y_true, pred, MARGINS)
        op5 = operating_point(miss, fa, warn, MARGINS, 0.05)
        op10 = operating_point(miss, fa, warn, MARGINS, 0.10)
        cover = float(np.mean(y_true <= pred))

        print(f"{name:<18}{mae(y_true, pred):>7.3f}{cover * 100:>8.1f} | "
              f"{fmt(op5, 0):>17}{fmt(op5, 2):>7}{fmt(op5, 1):>7} | "
              f"{fmt(op10, 0):>18}{fmt(op10, 2):>7}{fmt(op10, 1):>7}")

        row = {
            "horizon_h": H, "model": name, "scope": "all",
            "mae": mae(y_true, pred), "coverage": cover,
            "n_test": len(y_true), "n_below_threshold": n_low,
        }
        for tag, op in (("miss5", op5), ("miss10", op10)):
            row[f"{tag}_margin"] = op[0] if op else np.nan
            row[f"{tag}_false_alarm"] = op[1] if op else np.nan
            row[f"{tag}_warn_rate"] = op[2] if op else np.nan
        summary_rows.append(row)

        for mg, ms, fs, ws in zip(MARGINS, miss, fa, warn):
            curve_rows.append({"horizon_h": H, "model": name, "scope": "all",
                               "margin": mg, "miss": ms,
                               "false_alarm": fs, "warn_rate": ws})

        # Evening decisions only (8pm-style)
        if evening.any():
            miss_e, fa_e, warn_e, n_low_e = operating_curve(
                y_true[evening], pred[evening], MARGINS)
            op5e = operating_point(miss_e, fa_e, warn_e, MARGINS, 0.05)
            op10e = operating_point(miss_e, fa_e, warn_e, MARGINS, 0.10)
            erow = {
                "horizon_h": H, "model": name, "scope": "evening",
                "mae": mae(y_true[evening], pred[evening]),
                "coverage": float(np.mean(y_true[evening] <= pred[evening])),
                "n_test": int(evening.sum()), "n_below_threshold": n_low_e,
            }
            for tag, op in (("miss5", op5e), ("miss10", op10e)):
                erow[f"{tag}_margin"] = op[0] if op else np.nan
                erow[f"{tag}_false_alarm"] = op[1] if op else np.nan
                erow[f"{tag}_warn_rate"] = op[2] if op else np.nan
            summary_rows.append(erow)

    out = pd.DataFrame({"timestamp": test_h["timestamp"], "actual": y_true})
    for name, pred in predictions.items():
        out[name] = pred
    out.to_csv(OUT_DIR / f"safe_predictions_{H}h.csv", index=False)

# =============================================================================
# SAVE + FINAL SUMMARY
# =============================================================================

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "safe_forecast_summary.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT_DIR / "operating_curves.csv", index=False)

allrows = summary[summary["scope"] == "all"]

print("\n" + "=" * 80)
print("FALSE-ALARM RATE (%) NEEDED TO KEEP MISSES <= 5%   (lower is better)")
print("=" * 80)
fa5 = allrows.pivot(index="model", columns="horizon_h",
                    values="miss5_false_alarm") * 100
print(fa5.round(1).to_string(na_rep="n/a"))

print("\nSAFETY MARGIN (C) NEEDED TO KEEP MISSES <= 5%")
mg5 = allrows.pivot(index="model", columns="horizon_h", values="miss5_margin")
print(mg5.round(2).to_string(na_rep="n/a"))

print("\nCALIBRATION: share of actual values at or below the prediction (%)")
print("(Q05 should be ~5, Q10 ~10, Q20 ~20, Q50 ~50)")
cov = allrows.pivot(index="model", columns="horizon_h", values="coverage") * 100
print(cov.round(1).to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = pd.DataFrame(curve_rows)
    curves = curves[curves["scope"] == "all"]
    n = len(HORIZONS_H)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
    for ax, H in zip(axes[0], HORIZONS_H):
        for name, g in curves[curves["horizon_h"] == H].groupby("model"):
            ax.plot(g["false_alarm"] * 100, g["miss"] * 100, label=name)
        ax.set_title(f"{H} h ahead")
        ax.set_xlabel("False alarms (%)")
        ax.set_ylabel("Missed warnings (%)")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 60)
        ax.axhline(5, color="grey", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safety_tradeoff.png", dpi=150)
    print("\nSaved plot: safety_tradeoff.png")
except Exception:
    print("\n(matplotlib not available - skipped plot)")

print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
print(f"Saved to: {OUT_DIR}")
