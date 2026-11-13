"""
STMaCS SYSTEM 455 - FINAL MODELS v2 (calibrate-then-ship, no retraining)

Why v2 exists
-------------
v1 chose safety margins on a model trained on 80% of the training data, then
retrained on 100% and shipped it with the same margins. The honest check
showed that broke the guarantee: missed warnings rose from about 4% to 10-15%.
A margin is only valid for the exact fitted model it was measured on.

Procedure (this script)
-----------------------
  1. Combine the train and test periods into one timeline.
  2. FIT window        : the earliest 80% of that timeline. Q10 and Q50
                         anchored models are trained here, and ONLY here.
  3. CALIBRATION window: the most recent 20%. The safety margin is chosen
                         here: the smallest margin keeping missed warnings at
                         or below MISS_TARGET.
  4. The SAME fitted models are saved with their margins. Nothing is retrained
     after calibration.
  5. Stability check: the calibration window is cut into three consecutive
     chunks and the miss rate at the chosen margin is reported per chunk, to
     show whether the margin holds across sub-periods or only on average.

Limits to state in the report
-----------------------------
  - No unseen data remains after calibration. The evidence that the procedure
    generalises is the earlier held-out run (validated_margins), where it kept
    misses near 4% on a later test period.
  - The v1 retrain-after-calibration experiment showed the guarantee can break
    when the procedure is not followed. Recalibrate whenever you retrain.
  - Single installation, about one year of data.

Outputs (models/stmacs_v2/ unless --out-dir is given)
-----------------------------------------------------
  forecaster_1h.joblib ... forecaster_8h.joblib, config.json, model_card.json,
  calibration_summary.csv
These load with stmacs_forecaster.py exactly like v1.

Usage
-----
  python train_final_models_v2.py               full run
  python train_final_models_v2.py --quick       fast smoke test
  python train_final_models_v2.py --cushion 2.0 extra safety on top of margin
"""

import argparse
import datetime
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--ml-dir", type=str, default=None)
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--cushion", type=float, default=1.0)
parser.add_argument("--miss-target", type=float, default=0.05)
parser.add_argument("--calib-fraction", type=float, default=0.20)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = Path(args.ml_dir) if args.ml_dir else BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = (Path(args.out_dir) if args.out_dir
           else BASE_DIR / "models" / "stmacs_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_H = [1, 4] if args.quick else [1, 2, 4, 8]
MAX_ITER = 40 if args.quick else 300
CUSHION = float(args.cushion)
MISS_TARGET = float(args.miss_target)
CALIB_FRACTION = float(args.calib_fraction)
N_CHUNKS = 3

USABLE_TEMP = 45.0
MARGINS = np.arange(-2.0, 20.01, 0.25)
TOLERANCE_MIN = 3
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


def add_target(df, horizon_h):
    """Target looked up WITHIN this dataframe only."""
    df = df.copy()
    ts_ns = df["timestamp"].values.astype("datetime64[ns]").astype("int64")
    df["target_abs"] = lookup_nearest(
        ts_ns, df["buffer_top"].to_numpy(),
        ts_ns + horizon_h * NS_PER_HOUR, TOLERANCE_MIN * NS_PER_MIN)
    df["target_delta"] = df["target_abs"] - df["buffer_top"]
    return df.dropna(subset=["target_abs"]).reset_index(drop=True)


def make_quantile_model(q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q,
        max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def hour_anchor_table(df):
    hours = df["timestamp"].dt.hour.to_numpy()
    med = df.groupby(hours)["target_delta"].median()
    return np.array([med.get(h, 0.0) for h in range(24)])


def miss_curve(y_true, pred):
    actual_low = y_true < USABLE_TEMP
    n_low = int(actual_low.sum())
    if n_low == 0:
        return np.full(len(MARGINS), np.nan), 0
    warn = pred[:, None] < (USABLE_TEMP + MARGINS[None, :])
    return (actual_low[:, None] & ~warn).sum(axis=0) / n_low, n_low


def pick_margin(y_true, pred):
    miss, n_low = miss_curve(y_true, pred)
    if n_low == 0:
        raise RuntimeError("No low-temperature events in the calibration window")
    ok = np.where(miss <= MISS_TARGET)[0]
    if len(ok) == 0:
        return float(MARGINS[-1]), False, n_low
    return float(MARGINS[ok[0]]), True, n_low


def evaluate(y_true, pred, margin):
    actual_low = y_true < USABLE_TEMP
    warn = pred < (USABLE_TEMP + margin)
    n_low = int(actual_low.sum())
    n_ok = int((~actual_low).sum())
    miss = float((actual_low & ~warn).sum() / n_low) if n_low else float("nan")
    fa = float((~actual_low & warn).sum() / n_ok) if n_ok else float("nan")
    return miss, fa, n_low


def pct(x):
    return "   n/a" if x is None or np.isnan(x) else f"{x * 100:6.1f}"


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("FINAL MODELS v2 - CALIBRATE THEN SHIP" + ("  [QUICK]" if args.quick else ""))
print("=" * 80)

train_file = ML_DIR / "system455_60min_train.csv"
test_file = ML_DIR / "system455_60min_test.csv"
print(f"Train: {train_file}")
print(f"Test : {test_file}")
print(f"Output folder: {OUT_DIR}")

train_all = pd.read_csv(train_file)
test_all = pd.read_csv(test_file)
train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
test_all["timestamp"] = pd.to_datetime(test_all["timestamp"])

assert train_all["timestamp"].is_monotonic_increasing, "train not chronological"
assert test_all["timestamp"].is_monotonic_increasing, "test not chronological"
assert train_all["timestamp"].max() < test_all["timestamp"].min(), "overlap"

all_predictors = [c for c in train_all.columns
                  if c != "timestamp" and not c.startswith("target_")]
for name in REDUNDANT + CORE_EXTRA_DROP:
    if name not in all_predictors:
        raise ValueError(f"Column to drop not found in data: {name}")
FULL = [c for c in all_predictors if c not in REDUNDANT]
CORE = [c for c in FULL if c not in CORE_EXTRA_DROP]
print(f"\nFeature count: {len(CORE)} (expected 54)")
print(f"Miss target: {MISS_TARGET * 100:.0f}% | cushion: +{CUSHION:.1f} C | "
      f"calibration window: last {CALIB_FRACTION * 100:.0f}% of the timeline")

# =============================================================================
# BUILD
# =============================================================================

rows = []
bundles_info = {}
t_start = time.time()

for H in HORIZONS_H:
    print("\n" + "=" * 80)
    print(f"HORIZON: {H} hour(s) ahead")
    print("=" * 80)

    comb = pd.concat([add_target(train_all, H), add_target(test_all, H)],
                     ignore_index=True)
    comb = comb.sort_values("timestamp").reset_index(drop=True)

    cut = int(len(comb) * (1 - CALIB_FRACTION))
    cal_start = comb["timestamp"].iloc[cut]
    fit_h = comb.iloc[:cut]
    # Purge: drop fit rows whose target time falls inside the calibration window
    fit_h = fit_h[fit_h["timestamp"] + pd.Timedelta(hours=H) < cal_start]
    fit_h = fit_h.reset_index(drop=True)
    cal_h = comb.iloc[cut:].reset_index(drop=True)

    print(f"FIT        : {fit_h['timestamp'].min()} -> {fit_h['timestamp'].max()}"
          f"  ({len(fit_h):,} rows)")
    print(f"CALIBRATION: {cal_h['timestamp'].min()} -> {cal_h['timestamp'].max()}"
          f"  ({len(cal_h):,} rows)")

    anchor_tbl = hour_anchor_table(fit_h)
    anchor_fit = (fit_h["buffer_top"].to_numpy()
                  + anchor_tbl[fit_h["timestamp"].dt.hour.to_numpy()])
    resid_fit = fit_h["target_abs"].to_numpy() - anchor_fit

    t0 = time.time()
    q10 = make_quantile_model(0.10)
    q10.fit(fit_h[CORE], resid_fit)
    q50 = make_quantile_model(0.50)
    q50.fit(fit_h[CORE], resid_fit)
    print(f"  trained Q10 and Q50 on the FIT window ({time.time() - t0:.0f}s)")

    anchor_cal = (cal_h["buffer_top"].to_numpy()
                  + anchor_tbl[cal_h["timestamp"].dt.hour.to_numpy()])
    pred_cal = anchor_cal + q10.predict(cal_h[CORE])
    y_cal = cal_h["target_abs"].to_numpy()

    margin, reachable, n_events = pick_margin(y_cal, pred_cal)
    miss_c, fa_c, _ = evaluate(y_cal, pred_cal, margin)
    miss_cu, fa_cu, _ = evaluate(y_cal, pred_cal, margin + CUSHION)
    coverage = float(np.mean(y_cal <= pred_cal))

    print(f"\nCalibration window: {n_events:,} low-temperature events "
          f"of {len(y_cal):,} rows")
    print(f"  Chosen margin: {margin:+.2f} C"
          + ("" if reachable else "   [WARNING: target not reachable]"))
    print(f"  At margin       : miss {pct(miss_c)}%  false alarm {pct(fa_c)}%")
    print(f"  With +{CUSHION:.1f} C cushion: miss {pct(miss_cu)}%  "
          f"false alarm {pct(fa_cu)}%")
    print(f"  Calibration (Q10 should be ~10%): {coverage * 100:.1f}%")

    # Stability check: consecutive chunks of the calibration window
    chunk_edges = np.linspace(0, len(cal_h), N_CHUNKS + 1).astype(int)
    print(f"\n  Stability across {N_CHUNKS} consecutive chunks of the "
          "calibration window:")
    print(f"  {'chunk':<8}{'period start':<22}{'events':>8}"
          f"{'miss%':>8}{'FA%':>7}{'cal%':>7}")
    chunk_miss = []
    for c in range(N_CHUNKS):
        a, b = chunk_edges[c], chunk_edges[c + 1]
        ym, pm = y_cal[a:b], pred_cal[a:b]
        m_, f_, n_ = evaluate(ym, pm, margin)
        chunk_miss.append(m_)
        print(f"  {c + 1:<8}{str(cal_h['timestamp'].iloc[a])[:19]:<22}"
              f"{n_:>8,}{pct(m_):>8}{pct(f_):>7}"
              f"{float(np.mean(ym <= pm)) * 100:>7.1f}")
    valid_chunks = [m for m in chunk_miss if not np.isnan(m)]
    worst = max(valid_chunks) if valid_chunks else float("nan")
    if valid_chunks and worst > 2 * MISS_TARGET:
        print(f"  [CAUTION] worst chunk misses {worst * 100:.1f}% - the margin "
              "does not hold evenly across sub-periods. Consider a larger "
              "cushion for this horizon.")

    bundle = {
        "horizon_h": H,
        "features": CORE,
        "delta_by_hour": anchor_tbl.tolist(),
        "q10_model": q10,
        "q50_model": q50,
        "usable_temp": USABLE_TEMP,
        "margin": float(margin),
        "cushion": CUSHION,
        "n_training_rows": int(len(fit_h)),
        "training_start": str(fit_h["timestamp"].min()),
        "training_end": str(fit_h["timestamp"].max()),
        "calibration_start": str(cal_h["timestamp"].min()),
        "calibration_end": str(cal_h["timestamp"].max()),
    }
    joblib.dump(bundle, OUT_DIR / f"forecaster_{H}h.joblib", compress=3)
    bundles_info[H] = {
        "file": f"forecaster_{H}h.joblib",
        "margin": float(margin),
        "cushion": CUSHION,
        "warning_threshold_c": USABLE_TEMP + float(margin) + CUSHION,
        "n_training_rows": int(len(fit_h)),
    }
    print(f"\n  Saved: forecaster_{H}h.joblib")

    rows.append({
        "horizon_h": H, "margin": margin, "reachable": reachable,
        "cushion": CUSHION, "n_calibration_rows": len(y_cal),
        "n_calibration_events": n_events,
        "miss_at_margin": miss_c, "false_alarm_at_margin": fa_c,
        "miss_with_cushion": miss_cu, "false_alarm_with_cushion": fa_cu,
        "coverage_q10": coverage,
        "worst_chunk_miss": worst,
    })

# =============================================================================
# SAVE
# =============================================================================

config = {
    "name": "STMaCS safe forecaster v2 (calibrate-then-ship)",
    "horizons_h": HORIZONS_H,
    "usable_temp_c": USABLE_TEMP,
    "features": CORE,
    "warning_rule": "warn if pessimistic_top < usable_temp + margin + cushion",
    "bundles": bundles_info,
}
(OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

model_card = {
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__,
    "source_installation": "STMaCS system 455 (University of Botswana)",
    "procedure": ("fit on earliest 80% of train+test timeline; margin chosen on "
                  "most recent 20%; same fitted models shipped, no retraining"),
    "calibration_results": rows,
    "history": [
        "v1 retrained on 100% of training data after choosing margins; the "
        "honest check showed missed warnings rose from ~4% to 10-15%.",
        "Conclusion: margins are only valid for the exact fitted model; "
        "recalibrate whenever the model is retrained.",
    ],
    "caveats": [
        "No unseen data remains after calibration.",
        "Procedure evidence: earlier held-out run kept misses near 4% on test.",
        "Single installation (455), about one year of data.",
        "No weather inputs yet; solar gain inferred from collector/tank state.",
        "Heater effect is NOT modelled here; use physics in the controller.",
        "12-hour horizon dropped: no benefit over the hour-of-day baseline.",
    ],
}
(OUT_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))
pd.DataFrame(rows).to_csv(OUT_DIR / "calibration_summary.csv", index=False)

print("\n" + "=" * 80)
print("SUMMARY - CALIBRATION WINDOW (most recent data)")
print("=" * 80)
print(f"\n{'Horizon':<9}{'Margin':>8}{'Warn if <':>11}{'events':>8}"
      f"{'miss%':>8}{'FA%':>7}{'cal%':>7}{'worst chunk miss%':>19}")
for r in rows:
    print(f"{r['horizon_h']:<9}{r['margin']:>+8.2f}"
          f"{USABLE_TEMP + r['margin'] + CUSHION:>11.2f}"
          f"{r['n_calibration_events']:>8,}"
          f"{pct(r['miss_at_margin']):>8}{pct(r['false_alarm_at_margin']):>7}"
          f"{r['coverage_q10'] * 100:>7.1f}"
          f"{pct(r['worst_chunk_miss']):>19}")

print("\nHow to read this:")
print("  - 'miss%' should be at or below the target by construction. It is NOT")
print("    an honest test, because the margin was chosen on this same window.")
print("  - 'worst chunk miss%' is the useful number: it shows how far the miss")
print("    rate drifts between sub-periods. If it is much larger than the target,")
print("    add cushion (--cushion 2.0 or 3.0) for that horizon.")

print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
print(f"Saved to: {OUT_DIR}")
