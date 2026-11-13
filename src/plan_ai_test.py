"""
STMaCS - PLAN + SAFETY NET UNDER A FORECAST SURPRISE

Question answered
-----------------
The weather-aware planner saves energy when the sun forecast is right, but the
what-if test showed it can fail when the sun is far below forecast. Can a
short-horizon safety layer that watches the real sensors fix that, and is the
AI model better at that job than a trivial rule?

What it runs (digital twin, real-time scale, 3 days, 5 kW heater, 150 L/day)
---------------------------------------------------------------------------
Two cases:
  Forecast correct : the forecast and the real weather agree
  Forecast surprise: the forecast says clear, the real weather is overcast

Four strategies in each case:
  Thermostat 48 C       : plain thermostat on the middle sensor (S2)
  Plan only             : YOUR planner (loaded from predictive_control.py),
                          replanned every 30 minutes from the real tank state,
                          but always trusting the (possibly wrong) forecast
  Plan + simple S3 rule : the plan, plus a trivial rule: heater on if the tank
                          top (S3) is below 47 C, off again at 49 C
  Plan + AI safety net  : the plan, plus the AI early-warning controller
                          (1 h and 2 h pessimistic forecasts from the
                          trained twin models)

Measured: backup energy (kWh), litres of hot water delivered below the comfort
limit (S3 under 45 C), hours below comfort, heater switch-overs.

Run from the project folder:
    python src\\plan_ai_test.py                 both cases (about 10 minutes)
    python src\\plan_ai_test.py --quick         1 day, 1 case, hourly replanning
    python src\\plan_ai_test.py --case surprise
Options: --model-dir, --page-path (if predictive_control.py is not found),
         --replan-min, --heater-w, --thermostat-c
"""

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ai_closed_loop import (                                       # noqa: E402
    Scenario, make_inputs, _driven_twin_class, FORCE_ON_SETPOINT_C)
from utils.stmacs_features import FeatureBuilder, InsufficientHistory     # noqa: E402
from utils.stmacs_forecaster import SafeForecaster                        # noqa: E402
from utils.stmacs_controller import Controller, ControllerConfig, HeaterModel  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models" / "twin_v1"))
parser.add_argument("--page-path", default=None)
parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "twin_ml"))
parser.add_argument("--replan-min", type=int, default=30)
parser.add_argument("--heater-w", type=float, default=5000.0)
parser.add_argument("--thermostat-c", type=float, default=48.0)
parser.add_argument("--case", choices=["both", "correct", "surprise"], default="both")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
if args.quick:
    args.replan_min = max(args.replan_min, 60)
    args.case = "surprise"

COMFORT_C = 45.0
RULE_ON_C, RULE_OFF_C = 47.0, 49.0
START = datetime(2026, 7, 6)                      # a winter Monday
N_DAYS = 1 if args.quick else 3


# =============================================================================
# LOAD YOUR PLANNER
# =============================================================================

def load_page():
    if args.page_path:
        candidates = [Path(args.page_path)]
    else:
        candidates = [p for p in PROJECT_ROOT.rglob("predictive_control.py")
                      if ".venv" not in p.parts and "site-packages" not in p.parts]
    if not candidates:
        raise FileNotFoundError("predictive_control.py not found; pass --page-path")
    path = candidates[0]
    spec = importlib.util.spec_from_file_location("predictive_control_page", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


pc, page_path = load_page()
print("=" * 78)
print("PLAN + SAFETY NET UNDER A FORECAST SURPRISE" + ("  [QUICK]" if args.quick else ""))
print("=" * 78)
print(f"Planner loaded from: {page_path}")
print(f"AI models          : {args.model_dir}")
print(f"{N_DAYS} day(s); replanning every {args.replan_min} min; heater {args.heater_w:.0f} W")

forecaster = SafeForecaster(args.model_dir)

cfg = pc.Settings(
    source=pc.SRC_MANUAL, manual=(25.0, 38.0, 42.0, 30.0), clock_offset_h=0.0,
    weather_source=pc.WX_SYNTH, lat=0.0, lon=0.0, cloud_today=20.0, cloud_tomorrow=40.0,
    peak_irradiance=900.0, uncertainty=0.30, daily_use_l=150.0, morning=(6, 9),
    evening=(18, 21), comfort_min_c=COMFORT_C, heater_power_w=args.heater_w,
    heater_target_c=55.0, heater_inhibit_c=80.0, thermostat_c=args.thermostat_c,
    price_per_kwh=2.0, risk_tolerance=0.10, predictor=pc.PRED_TWIN,
)


def make_forecast(weather):
    sc = Scenario(start=START, weather=tuple(weather) + ("clear",) * 3, seed=7)
    irr, amb, _, _ = make_inputs(sc)
    times = [START + timedelta(hours=h) for h in range(len(irr) // 60)]
    return pc.Forecast(times, irr[::60].astype(float), amb[::60].astype(float),
                       "test forecast", "")


def make_plan(twin, now, forecast, predictor):
    start = pc._floor(now, 5)
    end = datetime.combine(start.date() + timedelta(days=2), dtime(0))
    steps = int((end - start).total_seconds() // pc.STEP_S)
    wins = pc.comfort_windows(cfg, start, end)
    snap = pc.Snapshot(np.array(twin.node_temperatures, dtype=float),
                       float(twin.collector_temperature), float(twin.middle_temperature),
                       float(twin.top_temperature), float(twin.bottom_temperature), "test")
    return pc.plan_backup(predictor, cfg, snap, forecast, start, steps, wins)


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(name, actual, forecast):
    irr, amb, cold, draw = actual
    n = len(irr)
    Twin = _driven_twin_class()
    twin = Twin(initial_tank_temperature=50.0, ambient_temperature=float(amb[0]),
                cold_water_temperature=float(cold[0]), heater_power=0.0)
    twin.node_temperatures = np.linspace(55.0, 40.0, twin.tank_nodes)
    twin.collector_temperature = 30.0
    twin.update_derived_state()

    uses_plan = name != "Thermostat"
    uses_ai = name.endswith("AI safety net")
    uses_rule = name.endswith("simple S3 rule")
    predictor = pc.TwinPredictor()
    rng = np.random.default_rng(1234)
    fb = FeatureBuilder() if uses_ai else None
    ctrl = (Controller(HeaterModel(power_kw=args.heater_w / 1000.0, volume_l=300.0),
                       ControllerConfig(extra_cushion={2: 2.0})) if uses_ai else None)

    windows, ai_on, rule_on = [], False, False
    energy_wh = cold_l = below_min = cold_win = 0.0
    switches, last_run = 0, False
    windows_h = [range(cfg.morning[0], cfg.morning[1]), range(cfg.evening[0], cfg.evening[1])]

    for k in range(n):
        now = START + timedelta(minutes=k)
        if uses_plan and k % args.replan_min == 0:
            windows = make_plan(twin, now, forecast, predictor).windows

        twin._irr_now = float(irr[k])
        twin.set_ambient_temperature(amb[k])
        twin.cold_water_temperature = float(cold[k])
        twin.set_hot_water_demand(draw[k] / 60.0)

        if name == "Thermostat":
            twin.set_heater(args.heater_w, args.thermostat_c)
        elif twin.top_temperature >= cfg.heater_inhibit_c:
            twin.set_heater(0.0, cfg.heater_target_c)
        elif (uses_ai and ai_on) or (uses_rule and rule_on):
            twin.set_heater(args.heater_w, FORCE_ON_SETPOINT_C)
        elif any(w.start <= now < w.end for w in windows):
            twin.set_heater(args.heater_w, cfg.heater_target_c)
        else:
            twin.set_heater(0.0, cfg.heater_target_c)
        twin.step(60)

        running = twin.heater_on and twin.heater_power > 0
        if running:
            energy_wh += twin.heater_power / 60.0
        if twin.top_temperature < COMFORT_C:
            below_min += 1
            cold_l += draw[k]
            if any(now.hour in r for r in windows_h):
                cold_win += draw[k]
        if running != last_run:
            switches += 1
            last_run = running

        if (k + 1) % 5 == 0:
            tnow = START + timedelta(minutes=k + 1)
            top, mid, bot = twin.top_temperature, twin.middle_temperature, twin.bottom_temperature
            noisy = [round(v + rng.normal(0, 0.1), 1)
                     for v in (top, mid, bot, twin.collector_temperature)]
            if uses_ai:
                fb.update(tnow, noisy[0], noisy[1], noisy[2], noisy[3],
                          twin.flow_rate * 60.0, int(twin.pump_on))
                try:
                    fcast = forecaster.predict(fb.features(), hour=tnow.hour,
                                               horizons=(1, 2), expected=False)
                except InsufficientHistory:
                    fcast = None
                ai_on = ctrl.step(tnow, noisy[0], fcast, buffer_centre=noisy[1]).heater_on
            if uses_rule:
                if noisy[0] < RULE_ON_C:
                    rule_on = True
                elif noisy[0] >= RULE_OFF_C:
                    rule_on = False

    return {"backup_kwh": round(energy_wh / 1000.0, 2),
            "cold_L_in_windows": round(float(cold_win), 1),
            "cold_L_other_hours": round(float(cold_l - cold_win), 1),
            "hours_below": round(below_min / 60.0, 1),
            "switch_overs": switches}


CASES = {
    "correct": ("Forecast correct", ("clear", "partly", "clear")[:N_DAYS],
                ("clear", "partly", "clear")[:N_DAYS]),
    "surprise": ("Forecast surprise (forecast clear, actual overcast)",
                 ("clear",) * N_DAYS, ("overcast",) * N_DAYS),
}
STRATEGIES = ["Thermostat", "Plan only", "Plan + simple S3 rule", "Plan + AI safety net"]

rows = []
t0 = time.time()
for key in ("correct", "surprise"):
    if args.case not in ("both", key):
        continue
    label, fc_weather, actual_weather = CASES[key]
    actual = make_inputs(Scenario(start=START, weather=actual_weather, daily_use_l=150.0, seed=7))
    forecast = make_forecast(fc_weather)
    print(f"\n--- {label} ---", flush=True)
    for name in STRATEGIES:
        m = simulate(name, actual, forecast)
        shown = f"Thermostat {args.thermostat_c:.0f} C" if name == "Thermostat" else name
        rows.append({"case": label, "strategy": shown, **m})
        print(f"{shown:<24} {m['backup_kwh']:>6} kWh | cold L: {m['cold_L_in_windows']:>5} in demand "
              f"windows, {m['cold_L_other_hours']:>5} other hours | {m['switch_overs']:>3} switch-overs "
              f"[{time.time() - t0:.0f}s]", flush=True)

df = pd.DataFrame(rows)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "plan_ai_test.csv", index=False)
print("\n" + "=" * 78)
print("RESULTS")
print("=" * 78)
print(df.to_string(index=False))
print("\nHOW TO READ THIS")
print("- 'cold L' is hot water drawn while the tank top was below 45 C. The planner only")
print("  promises comfort in the demand windows (6-9 and 18-21), so judge it on 'in_windows';")
print("  the thermostat protects every hour, which is a stronger service level.")
print("- In the surprise case, compare 'Plan only' with the two safety layers: does either")
print("  remove the cold water, and at what energy cost?")
print("- The AI is only worth including if it beats the simple S3 rule. Report what you see.")
print(f"\nSaved: {out / 'plan_ai_test.csv'}")
print(f"Total time: {(time.time() - t0) / 60:.1f} min")