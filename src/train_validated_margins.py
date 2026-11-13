"""
STMaCS SYSTEM 455 - VALIDATED SAFETY MARGINS FOR THE CONTROLLER

Why this script exists
----------------------
The previous run picked each safety margin on the SAME test data it then
reported on, which flatters the results. This script does it properly:

  1. The training period is split chronologically into FIT (first 80%) and
     VALIDATION (last 20%). Models are trained on FIT only.
  2. Safety margins are chosen on VALIDATION: "the smallest margin that keeps
     missed warnings at or below MISS_TARGET".
  3. Those frozen margins are then applied to the untouched TEST period, and we
     report what actually happened. The gap between the validation miss rate
     and the test miss rate shows how well the margin carries over to a new
     season. Extra "cushion" values (+1, +2 degrees) show how much more
     margin would close that gap.

It also reports results by time-of-day window, because the controller can run
at any hour and forecast reliability differs between night, morning, midday,
afternoon and evening.

Models per horizon (CORE: the 54 portable features)
---------------------------------------------------
  Persistence, Hour-of-day      baselines
  Q50 / Q10 / Q05 anchored      quantile models on the hour-of-day residual

Two margin policies
-------------------
  global      : one margin per horizon and model (chosen on all validation rows)
  per-window  : one margin per horizon, model and time window; falls back to
                the global margin where validation has fewer than MIN_EVENTS
                low-temperature events (too few to trust)

Outputs (in baseline_results/validated_margins/)
------------------------------------------------
  validated_margins.csv          the margins chosen on validation
  test_results.csv               what those margins did on the test period
  controller_table.csv           recommended model + margin per horizon/window

Usage
-----
  python train_validated_margins.py              full run (~8 min)
  python train_validated_margins.py --quick      fast smoke test
  python train_validated_margins.py --ml-dir PATH
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
parser.add_argument("--ml-dir", type=str, default=None)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = Path(args.ml_dir) if args.ml_dir else BASE_DIR / "data" / "processed" / "ml"
OUT_DIR = ML_DIR / "baseline_results" / "validated_margins"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_H = [1, 4] if args.quick else [1, 2, 4, 8, 12]
MAX_ITER = 40 if args.quick else 300
QUANTILES = [0.5, 0.1, 0.05]

USABLE_TEMP = 45.0
MISS_TARGET = 0.05          # acceptable missed-warning rate
MIN_EVENTS = 30             # min validation events to trust a per-window margin
VAL_FRACTION = 0.20         # last 20% of the training period
CUSHIONS = [0.0, 1.0, 2.0]  # extra degrees added on top of validated margins
MARGINS = np.arange(-2.0, 20.01, 0.25)
TOLERANCE_MIN = 3
RANDOM_STATE = 42

# Time-of-day windows by DECISION hour: name -> (start_hour, end_hour_exclusive)
WINDOWS = {
    "night 00-05": (0, 5),
    "morning 05-09": (5, 9),
    "midday 09-15": (9, 15),
    "afternoon 15-18": (15, 18),
    "evening 18-24": (18, 24),
}

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
    """Target = buffer_top at timestamp + horizon, looked up WITHIN this
    dataframe only (so fit, validation and test never share targets)."""
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


def miss_curve(y_true, pred, margins):
    """Miss rate for every margin (vectorised). Also returns event count."""
    actual_low = y_true < USABLE_TEMP
    n_low = int(actual_low.sum())
    if n_low == 0:
        return np.full(len(margins), np.nan), 0
    warn = pred[:, None] < (USABLE_TEMP + margins[None, :])
    miss = (actual_low[:, None] & ~warn).sum(axis=0) / n_low
    return miss, n_low


def pick_margin(y_true, pred, target):
    """Smallest margin whose miss rate is <= target.
    Returns (margin, reachable, n_events). margin is NaN if no events."""
    miss, n_low = miss_curve(y_true, pred, MARGINS)
    if n_low == 0:
        return np.nan, False, 0
    ok = np.where(miss <= target)[0]
    if len(ok) == 0:
        return float(MARGINS[-1]), False, n_low
    return float(MARGINS[ok[0]]), True, n_low


def evaluate(y_true, pred, margin):
    """Miss, false-alarm, warn rate and counts at one fixed margin."""
    actual_low = y_true < USABLE_TEMP
    warn = pred < (USABLE_TEMP + margin)
    n_low = int(actual_low.sum())
    n_ok = int((~actual_low).sum())
    miss = float((actual_low & ~warn).sum() / n_low) if n_low else np.nan
    fa = float((~actual_low & warn).sum() / n_ok) if n_ok else np.nan
    return miss, fa, float(warn.mean()) if len(warn) else np.nan, n_low


def pct(x):
    return "   n/a" if x is None or np.isnan(x) else f"{x * 100:6.1f}"


# =============================================================================
# LOAD AND SPLIT
# =============================================================================

print("=" * 80)
print("VALIDATED SAFETY MARGINS" + ("  [QUICK]" if args.quick else ""))
print("=" * 80)

train_file = ML_DIR / "system455_60min_train.csv"
test_file = ML_DIR / "system455_60min_test.csv"
print(f"Train: {train_file}")
print(f"Test : {test_file}")

train_all = pd.read_csv(train_file)
test_raw = pd.read_csv(test_file)
train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
test_raw["timestamp"] = pd.to_datetime(test_raw["timestamp"])

assert train_all["timestamp"].is_monotonic_increasing, "train not chronological"
assert test_raw["timestamp"].is_monotonic_increasing, "test not chronological"
assert train_all["timestamp"].max() < test_raw["timestamp"].min(), "overlap"

cut = int(len(train_all) * (1 - VAL_FRACTION))
fit_raw = train_all.iloc[:cut].reset_index(drop=True)
val_raw = train_all.iloc[cut:].reset_index(drop=True)

print(f"\nFIT        : {fit_raw['timestamp'].min()}  ->  "
      f"{fit_raw['timestamp'].max()}  ({len(fit_raw):,} rows)")
print(f"VALIDATION : {val_raw['timestamp'].min()}  ->  "
      f"{val_raw['timestamp'].max()}  ({len(val_raw):,} rows)")
print(f"TEST       : {test_raw['timestamp'].min()}  ->  "
      f"{test_raw['timestamp'].max()}  ({len(test_raw):,} rows)")

all_predictors = [c for c in train_all.columns
                  if c != "timestamp" and not c.startswith("target_")]
for name in REDUNDANT + CORE_EXTRA_DROP:
    if name not in all_predictors:
        raise ValueError(f"Column to drop not found in data: {name}")
FULL = [c for c in all_predictors if c not in REDUNDANT]
CORE = [c for c in FULL if c not in CORE_EXTRA_DROP]
print(f"\nFeature count: {len(CORE)} (expected 54)")
print(f"Usable threshold: {USABLE_TEMP:.0f} C | miss target: "
      f"{MISS_TARGET * 100:.0f}% | min events per window: {MIN_EVENTS}")

# =============================================================================
# MAIN LOOP
# =============================================================================

margin_rows = []
test_rows = []
controller_rows = []
t_start = time.time()

for H in HORIZONS_H:
    print("\n" + "=" * 80)
    print(f"HORIZON: {H} hour(s) ahead")
    print("=" * 80)

    fit_h = add_target(fit_raw, H)
    val_h = add_target(val_raw, H)
    test_h = add_target(test_raw, H)
    print(f"Rows with valid target - fit: {len(fit_h):,}  "
          f"validation: {len(val_h):,}  test: {len(test_h):,}")

    # Hour-of-day anchor, from FIT data only
    fit_hour = fit_h["timestamp"].dt.hour.to_numpy()
    hour_delta = fit_h.groupby(fit_hour)["target_delta"].median()
    delta_by_hour = np.array([hour_delta.get(h, 0.0) for h in range(24)])

    def anchor(df):
        return (df["buffer_top"].to_numpy()
                + delta_by_hour[df["timestamp"].dt.hour.to_numpy()])

    anchor_fit, anchor_val, anchor_test = anchor(fit_h), anchor(val_h), anchor(test_h)
    resid_fit = fit_h["target_abs"].to_numpy() - anchor_fit

    preds_val = {"Persistence": val_h["buffer_top"].to_numpy(),
                 "Hour-of-day": anchor_val}
    preds_test = {"Persistence": test_h["buffer_top"].to_numpy(),
                  "Hour-of-day": anchor_test}

    for q in QUANTILES:
        t0 = time.time()
        model = make_quantile_model(q)
        model.fit(fit_h[CORE], resid_fit)
        label = f"Q{int(q * 100):02d} anchored"
        preds_val[label] = anchor_val + model.predict(val_h[CORE])
        preds_test[label] = anchor_test + model.predict(test_h[CORE])
        print(f"  trained {label}   ({time.time() - t0:.0f}s)")

    y_val = val_h["target_abs"].to_numpy()
    y_test = test_h["target_abs"].to_numpy()
    hour_val = val_h["timestamp"].dt.hour.to_numpy()
    hour_test = test_h["timestamp"].dt.hour.to_numpy()

    win_val = {n: (hour_val >= a) & (hour_val < b) for n, (a, b) in WINDOWS.items()}
    win_test = {n: (hour_test >= a) & (hour_test < b) for n, (a, b) in WINDOWS.items()}

    print(f"\nLow-temperature events (target below {USABLE_TEMP:.0f} C): "
          f"validation {int((y_val < USABLE_TEMP).sum()):,}, "
          f"test {int((y_test < USABLE_TEMP).sum()):,}")

    # ------------------------------------------------------------------
    # Choose margins on VALIDATION, then evaluate on TEST
    # ------------------------------------------------------------------
    print(f"\n{'Model':<16}{'valMargin':>10}{'valMiss%':>9}{'valFA%':>8} | "
          f"{'testMiss%':>10}{'testFA%':>9}{'+1C miss%':>10}{'+2C miss%':>10}"
          f"{'  coverage val/test':>20}")

    horizon_candidates = {}

    for name in preds_val:
        g_margin, g_ok, g_events = pick_margin(y_val, preds_val[name], MISS_TARGET)
        v_miss, v_fa, _, _ = evaluate(y_val, preds_val[name], g_margin)

        # Test results at the frozen global margin (+ cushions)
        t_by_cushion = {}
        for c in CUSHIONS:
            m_, f_, w_, n_ = evaluate(y_test, preds_test[name], g_margin + c)
            t_by_cushion[c] = (m_, f_, w_, n_)
            test_rows.append({
                "horizon_h": H, "model": name, "policy": "global",
                "window": "all", "cushion": c, "margin": g_margin + c,
                "n_test": len(y_test), "n_events": n_,
                "miss_rate": m_, "false_alarm": f_, "warn_rate": w_,
            })

        cov_v = float(np.mean(y_val <= preds_val[name]))
        cov_t = float(np.mean(y_test <= preds_test[name]))
        print(f"{name:<16}{g_margin:>+10.2f}{pct(v_miss):>9}{pct(v_fa):>8} | "
              f"{pct(t_by_cushion[0.0][0]):>10}{pct(t_by_cushion[0.0][1]):>9}"
              f"{pct(t_by_cushion[1.0][0]):>10}{pct(t_by_cushion[2.0][0]):>10}"
              f"{cov_v * 100:>12.1f}/{cov_t * 100:<5.1f}")

        margin_rows.append({
            "horizon_h": H, "model": name, "window": "all",
            "margin": g_margin, "reachable": g_ok, "n_val_events": g_events,
            "val_miss": v_miss, "val_false_alarm": v_fa,
        })

        # Per-window margins (fallback to global when too few events)
        win_margin = {}
        for wname, vmask in win_val.items():
            wm, w_ok, w_n = pick_margin(
                y_val[vmask], preds_val[name][vmask], MISS_TARGET)
            used = wm if (w_n >= MIN_EVENTS and not np.isnan(wm)) else g_margin
            source = "window" if used == wm and w_n >= MIN_EVENTS else "global"
            win_margin[wname] = (used, source, w_n)
            margin_rows.append({
                "horizon_h": H, "model": name, "window": wname,
                "margin": used, "reachable": w_ok, "n_val_events": w_n,
                "val_miss": np.nan, "val_false_alarm": np.nan,
                "source": source,
            })

        for c in CUSHIONS:
            for wname, tmask in win_test.items():
                used, source, _ = win_margin[wname]
                m_, f_, w_, n_ = evaluate(
                    y_test[tmask], preds_test[name][tmask], used + c)
                test_rows.append({
                    "horizon_h": H, "model": name, "policy": "per-window",
                    "window": wname, "cushion": c, "margin": used + c,
                    "n_test": int(tmask.sum()), "n_events": n_,
                    "miss_rate": m_, "false_alarm": f_, "warn_rate": w_,
                })
                # global policy by window, for a like-for-like comparison
                m2, f2, w2, n2 = evaluate(
                    y_test[tmask], preds_test[name][tmask], g_margin + c)
                test_rows.append({
                    "horizon_h": H, "model": name, "policy": "global",
                    "window": wname, "cushion": c, "margin": g_margin + c,
                    "n_test": int(tmask.sum()), "n_events": n2,
                    "miss_rate": m2, "false_alarm": f2, "warn_rate": w2,
                })

        horizon_candidates[name] = (g_margin, v_fa, g_ok, win_margin)

    # ------------------------------------------------------------------
    # Recommended model for this horizon: lowest VALIDATION false-alarm rate
    # among models that reach the miss target on validation
    # ------------------------------------------------------------------
    reachable = {n: v for n, v in horizon_candidates.items() if v[2]}
    pool = reachable if reachable else horizon_candidates
    best = min(pool, key=lambda n: pool[n][1])
    g_margin, _, _, win_margin = horizon_candidates[best]

    print(f"\nRECOMMENDED for {H} h: {best}  (chosen on validation, global margin "
          f"{g_margin:+.2f} C)")
    print(f"\n{'Window':<18}{'n':>7}{'events':>8} | "
          f"{'global: miss%':>14}{'FA%':>7} | "
          f"{'per-window: margin':>19}{'src':>8}{'miss%':>7}{'FA%':>7}")

    for wname, tmask in win_test.items():
        if not tmask.any():
            continue
        mg, fg, _, ng = evaluate(
            y_test[tmask], preds_test[best][tmask], g_margin)
        used, source, _ = win_margin[wname]
        mw, fw, _, _ = evaluate(
            y_test[tmask], preds_test[best][tmask], used)
        print(f"{wname:<18}{int(tmask.sum()):>7,}{ng:>8,} | "
              f"{pct(mg):>14}{pct(fg):>7} | "
              f"{used:>+19.2f}{source:>8}{pct(mw):>7}{pct(fw):>7}")

        controller_rows.append({
            "horizon_h": H, "window": wname, "model": best,
            "global_margin": g_margin, "window_margin": used,
            "window_margin_source": source,
            "n_test": int(tmask.sum()), "n_test_events": ng,
            "test_miss_global": mg, "test_fa_global": fg,
            "test_miss_window": mw, "test_fa_window": fw,
        })

# =============================================================================
# SAVE + FINAL SUMMARY
# =============================================================================

pd.DataFrame(margin_rows).to_csv(OUT_DIR / "validated_margins.csv", index=False)
test_df = pd.DataFrame(test_rows)
test_df.to_csv(OUT_DIR / "test_results.csv", index=False)
pd.DataFrame(controller_rows).to_csv(OUT_DIR / "controller_table.csv", index=False)

ctrl = pd.DataFrame(controller_rows)

print("\n" + "=" * 80)
print("SUMMARY: RECOMMENDED MODEL PER HORIZON, TEST PERIOD (all-day, global margin)")
print("=" * 80)
print(f"\n{'Horizon':<9}{'Model':<16}{'Margin':>8}"
      f"{'miss%':>8}{'FA%':>7}{'miss% +1C':>11}{'FA% +1C':>9}"
      f"{'miss% +2C':>11}{'FA% +2C':>9}")
for H in HORIZONS_H:
    row = ctrl[ctrl["horizon_h"] == H].iloc[0]
    model = row["model"]
    sub = test_df[(test_df["horizon_h"] == H) & (test_df["model"] == model)
                  & (test_df["policy"] == "global") & (test_df["window"] == "all")]
    vals = {r["cushion"]: r for _, r in sub.iterrows()}
    print(f"{H:<9}{model:<16}{row['global_margin']:>+8.2f}"
          f"{pct(vals[0.0]['miss_rate']):>8}{pct(vals[0.0]['false_alarm']):>7}"
          f"{pct(vals[1.0]['miss_rate']):>11}{pct(vals[1.0]['false_alarm']):>9}"
          f"{pct(vals[2.0]['miss_rate']):>11}{pct(vals[2.0]['false_alarm']):>9}")

print("\nHow to read this:")
print(f"  - The margin was chosen on VALIDATION to give a miss rate of "
      f"{MISS_TARGET * 100:.0f}% or less.")
print("  - 'miss%' on TEST is what really happened on unseen data. If it is "
      "well above the target, the test season is harder than validation.")
print("  - The +1C / +2C columns show how much extra margin closes the gap, "
      "and what it costs in false alarms.")

print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
print(f"Saved to: {OUT_DIR}")
