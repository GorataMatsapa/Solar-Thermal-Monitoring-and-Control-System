"""
STMaCS SYSTEM 455 - FINAL MODELS FOR THE DIGITAL TWIN / CONTROLLER

What this script does
---------------------
Takes the design that survived validation and produces the files the
controller will actually load.

Design (from the validation results)
------------------------------------
  - Horizons deployed: 1, 2, 4 and 8 hours (12 h was not useful and is dropped)
  - Safety model  : Q10 "anchored" quantile model (a pessimistic forecast)
  - Display model : Q50 "anchored" quantile model (the expected temperature)
  - Features      : the 54 portable CORE features
  - Margin        : one validated margin per horizon (read from
                    validated_margins.csv) plus a +CUSHION safety cushion
  - Warning rule  : warn if  pessimistic_forecast < USABLE_TEMP + margin + cushion

Two builds
----------
  1. EVALUATION BUILD: trains Q10 on the full TRAIN period only and checks the
     frozen margins on the untouched TEST period. This is the honest check that
     retraining on 100% of the training data did not break the margins.
     Nothing is re-tuned here.
  2. PRODUCTION BUILD: trains Q10 and Q50 on TRAIN + TEST together (all
     available data, including the most recent season) and saves them.
     Its test numbers would not be honest (it has seen the test data), so only
     build 1 is used for reporting.

Outputs (in models/stmacs_v1/ unless --out-dir is given)
--------------------------------------------------------
  forecaster_1h.joblib ... forecaster_8h.joblib   model bundles
  config.json                                      human-readable settings
  model_card.json                                  data, metrics, caveats

Use the saved models with stmacs_forecaster.py.

Usage
-----
  python train_final_models.py                 full run (~6 min)
  python train_final_models.py --quick         fast smoke test
  python train_final_models.py --cushion 2.0   use a +2 C cushion instead
  python train_final_models.py --ml-dir PATH --out-dir PATH
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
parser.add_argument("--cushion", type=float, default=1.0,
                    help="extra degrees C added on top of the validated margin")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = Path(args.ml_dir) if args.ml_dir else BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = (Path(args.out_dir) if args.out_dir
           else BASE_DIR / "models" / "stmacs_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARGINS_CSV = ML_DIR / "baseline_results" / "validated_margins" / "validated_margins.csv"

HORIZONS_H = [1, 4] if args.quick else [1, 2, 4, 8]
MAX_ITER = 40 if args.quick else 300
CUSHION = float(args.cushion)

USABLE_TEMP = 45.0
TOLERANCE_MIN = 3
RANDOM_STATE = 42

# Fallback margins (from the validation run) if validated_margins.csv is absent
DEFAULT_MARGINS = {1: -0.50, 2: -0.50, 4: 0.50, 8: 2.50}

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
    """Target looked up WITHIN this dataframe only (train and test never mix)."""
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
    """Median change over the horizon for each decision hour (0-23)."""
    hours = df["timestamp"].dt.hour.to_numpy()
    med = df.groupby(hours)["target_delta"].median()
    return np.array([med.get(h, 0.0) for h in range(24)])


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


def load_margins():
    margins = dict(DEFAULT_MARGINS)
    source = "built-in defaults"
    if MARGINS_CSV.exists():
        m = pd.read_csv(MARGINS_CSV)
        m = m[(m["model"] == "Q10 anchored") & (m["window"] == "all")]
        found = {int(r["horizon_h"]): float(r["margin"]) for _, r in m.iterrows()}
        margins.update(found)
        source = f"{MARGINS_CSV.name} (validation run)"
    return margins, source


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("FINAL MODEL BUILD" + ("  [QUICK]" if args.quick else ""))
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

margins, margin_source = load_margins()
print(f"Validated margins loaded from: {margin_source}")
print(f"Cushion: +{CUSHION:.2f} C")
print(f"Usable threshold: {USABLE_TEMP:.0f} C at buffer_top")

# =============================================================================
# BUILD
# =============================================================================

eval_rows = []
t_start = time.time()
bundles_info = {}

for H in HORIZONS_H:
    print("\n" + "=" * 80)
    print(f"HORIZON: {H} hour(s) ahead   (validated margin "
          f"{margins[H]:+.2f} C, with cushion {margins[H] + CUSHION:+.2f} C)")
    print("=" * 80)

    train_h = add_target(train_all, H)
    test_h = add_target(test_all, H)

    # ---------------- 1. EVALUATION BUILD (train only) -----------------
    t0 = time.time()
    anchor_tbl_eval = hour_anchor_table(train_h)
    anchor_train = (train_h["buffer_top"].to_numpy()
                    + anchor_tbl_eval[train_h["timestamp"].dt.hour.to_numpy()])
    resid_train = train_h["target_abs"].to_numpy() - anchor_train

    eval_model = make_quantile_model(0.10)
    eval_model.fit(train_h[CORE], resid_train)

    anchor_test = (test_h["buffer_top"].to_numpy()
                   + anchor_tbl_eval[test_h["timestamp"].dt.hour.to_numpy()])
    pred_test = anchor_test + eval_model.predict(test_h[CORE])
    y_test = test_h["target_abs"].to_numpy()

    miss0, fa0, n_low = evaluate(y_test, pred_test, margins[H])
    missc, fac, _ = evaluate(y_test, pred_test, margins[H] + CUSHION)
    coverage = float(np.mean(y_test <= pred_test))

    print(f"\nEVALUATION BUILD (trained on train only, {len(train_h):,} rows, "
          f"{time.time() - t0:.0f}s)")
    print(f"  Test events below {USABLE_TEMP:.0f} C: {n_low:,} of {len(y_test):,}")
    print(f"  Validated margin {margins[H]:+.2f} C : "
          f"miss {pct(miss0)}%  false alarm {pct(fa0)}%")
    print(f"  With +{CUSHION:.1f} C cushion       : "
          f"miss {pct(missc)}%  false alarm {pct(fac)}%")
    print(f"  Calibration (Q10 should be ~10%): {coverage * 100:.1f}%")

    eval_rows.append({
        "horizon_h": H, "margin": margins[H], "cushion": CUSHION,
        "n_test": len(y_test), "n_test_events": n_low,
        "miss_at_margin": miss0, "false_alarm_at_margin": fa0,
        "miss_with_cushion": missc, "false_alarm_with_cushion": fac,
        "coverage_q10": coverage,
    })

    # ---------------- 2. PRODUCTION BUILD (train + test) ---------------
    t0 = time.time()
    prod_h = pd.concat([train_h, test_h], ignore_index=True)
    anchor_tbl = hour_anchor_table(prod_h)
    anchor_prod = (prod_h["buffer_top"].to_numpy()
                   + anchor_tbl[prod_h["timestamp"].dt.hour.to_numpy()])
    resid_prod = prod_h["target_abs"].to_numpy() - anchor_prod

    q10 = make_quantile_model(0.10)
    q10.fit(prod_h[CORE], resid_prod)
    q50 = make_quantile_model(0.50)
    q50.fit(prod_h[CORE], resid_prod)

    bundle = {
        "horizon_h": H,
        "features": CORE,
        "delta_by_hour": anchor_tbl.tolist(),
        "q10_model": q10,
        "q50_model": q50,
        "usable_temp": USABLE_TEMP,
        "margin": float(margins[H]),
        "cushion": CUSHION,
        "n_training_rows": int(len(prod_h)),
        "training_start": str(prod_h["timestamp"].min()),
        "training_end": str(prod_h["timestamp"].max()),
    }
    joblib.dump(bundle, OUT_DIR / f"forecaster_{H}h.joblib", compress=3)
    bundles_info[H] = {
        "file": f"forecaster_{H}h.joblib",
        "margin": float(margins[H]),
        "cushion": CUSHION,
        "warning_threshold_c": USABLE_TEMP + float(margins[H]) + CUSHION,
        "n_training_rows": int(len(prod_h)),
    }
    print(f"\nPRODUCTION BUILD (train + test, {len(prod_h):,} rows, "
          f"{time.time() - t0:.0f}s)")
    print(f"  Saved: forecaster_{H}h.joblib")

# =============================================================================
# SAVE CONFIG + MODEL CARD
# =============================================================================

config = {
    "name": "STMaCS safe forecaster v1",
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
    "train_period": [str(train_all["timestamp"].min()),
                     str(train_all["timestamp"].max())],
    "test_period": [str(test_all["timestamp"].min()),
                    str(test_all["timestamp"].max())],
    "production_trained_on": "train + test periods",
    "honest_evaluation": ("evaluation build: trained on train only, "
                          "margins frozen from validation, scored on test"),
    "evaluation_results": eval_rows,
    "caveats": [
        "Single installation (455), about one year of data, one winter test.",
        "Test season had fewer low-temperature events than validation.",
        "Consecutive 5-minute rows are highly correlated; events are fewer "
        "than rows suggest.",
        "No weather inputs yet; solar gain is inferred from collector/tank state.",
        "Heater effect is NOT modelled here; use physics in the controller.",
        "12-hour horizon dropped: ML added nothing over the hour-of-day baseline.",
    ],
}
(OUT_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))

pd.DataFrame(eval_rows).to_csv(OUT_DIR / "evaluation_summary.csv", index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY - HONEST CHECK (trained on train only, scored on test)")
print("=" * 80)
print(f"\n{'Horizon':<9}{'Margin':>8}{'miss%':>8}{'FA%':>7}"
      f"{'miss% +cush':>13}{'FA% +cush':>11}{'cal%':>7}{'events':>8}")
for r in eval_rows:
    print(f"{r['horizon_h']:<9}{r['margin']:>+8.2f}"
          f"{pct(r['miss_at_margin']):>8}{pct(r['false_alarm_at_margin']):>7}"
          f"{pct(r['miss_with_cushion']):>13}"
          f"{pct(r['false_alarm_with_cushion']):>11}"
          f"{r['coverage_q10'] * 100:>7.1f}{r['n_test_events']:>8,}")

print("\nHow to read this:")
print("  - These should match the earlier validated Q10 results closely.")
print("    A big jump in 'miss%' would mean retraining on all the training")
print("    data changed the model's calibration.")
print("  - The saved production models also learned from the test period,")
print("    so their own test scores would be inflated. Do not report those.")

print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
print(f"Saved to: {OUT_DIR}")
print("Files: forecaster_*h.joblib, config.json, model_card.json, "
      "evaluation_summary.csv")
