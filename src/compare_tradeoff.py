"""
STMaCS - THERMOSTAT vs AI CONTROLLER AT MATCHED COMFORT

Why this exists
---------------
Comparing the AI controller with a thermostat at ONE setting is unfair: each
strategy can trade energy for comfort. This script sweeps both:

  Thermostat : setpoint on the middle sensor (S2), 42 to 55 C
  AI         : extra safety margin on the 2-hour forecast, -2 to +4 C

over the same winter scenarios on the digital twin, and records for each
setting the backup energy used and the litres of hot water delivered below
the comfort limit (S3 under 45 C). Plotting energy against cold litres shows
which strategy gives better comfort for the same energy (or less energy for
the same comfort).

Run from the project folder:
    python src\\compare_tradeoff.py                       full (about 5-15 min)
    python src\\compare_tradeoff.py --quick               one scenario, fewer settings
    python src\\compare_tradeoff.py --model-dir models\\twin_v1

Outputs (data\\processed\\twin_ml\\):
    tradeoff_comparison.csv, tradeoff_comparison.png (if matplotlib is installed)
"""

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ai_closed_loop import Scenario, make_inputs, _simulate      # noqa: E402
from utils.stmacs_controller import ControllerConfig                     # noqa: E402
from utils.stmacs_forecaster import SafeForecaster                       # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "twin_v1"))
parser.add_argument("--out-dir", type=str,
                    default=str(PROJECT_ROOT / "data" / "processed" / "twin_ml"))
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

W = datetime(2026, 7, 6)
SCENARIOS = [
    Scenario(start=W, weather=("clear", "overcast", "partly"), seed=7),
    Scenario(start=W, weather=("overcast", "overcast", "overcast"), daily_use_l=200.0, seed=5),
    Scenario(start=datetime(2026, 7, 20), weather=("partly", "overcast", "clear"),
             daily_use_l=200.0, seed=11),
]
THERMOSTAT_SETPOINTS = [42, 44, 46, 48, 50, 52, 55]
AI_MARGINS = [-2, -1, 0, 1, 2, 3, 4]
if args.quick:
    SCENARIOS = SCENARIOS[:1]
    THERMOSTAT_SETPOINTS = [44, 48, 52]
    AI_MARGINS = [-1, 1, 3]

print("=" * 78)
print("THERMOSTAT vs AI CONTROLLER AT MATCHED COMFORT" + ("  [QUICK]" if args.quick else ""))
print("=" * 78)
print(f"Models: {args.model_dir}")
print(f"{len(SCENARIOS)} scenario(s) x 3 days each; comfort limit S3 >= 45 C")

forecaster = SafeForecaster(args.model_dir)
inputs = [make_inputs(s) for s in SCENARIOS]


def run(strategy, make_scenario, make_cfg):
    energy = cold = switches = 0.0
    for scen, inp in zip(SCENARIOS, inputs):
        _, m, _ = _simulate(strategy, make_scenario(scen), inp, forecaster, make_cfg(scen))
        energy += m["Backup energy (kWh)"]
        cold += m["Hot water drawn below comfort (L)"]
        switches += m["Heater switch-overs"]
    return round(energy, 1), round(float(cold), 1), int(switches)


rows = []
t0 = time.time()

e, c, s = run("No backup", lambda sc: sc, lambda sc: ControllerConfig())
rows.append({"strategy": "No backup", "setting": "-", "energy_kwh": e,
             "cold_litres": c, "switch_overs": s})
print(f"No backup: {e} kWh, {c} L cold  [{time.time() - t0:.0f}s]", flush=True)

for sp in THERMOSTAT_SETPOINTS:
    e, c, s = run("Thermostat", lambda sc, sp=sp: replace(sc, thermostat_c=float(sp)),
                  lambda sc: ControllerConfig())
    rows.append({"strategy": "Thermostat", "setting": f"{sp} C", "energy_kwh": e,
                 "cold_litres": c, "switch_overs": s})
    print(f"Thermostat {sp} C: {e} kWh, {c} L cold, {s} switch-overs  [{time.time() - t0:.0f}s]",
          flush=True)

for margin in AI_MARGINS:
    e, c, s = run("AI controller", lambda sc: sc,
                  lambda sc, m=margin: ControllerConfig(extra_cushion={2: float(m)}))
    rows.append({"strategy": "AI controller", "setting": f"{margin:+d} C margin",
                 "energy_kwh": e, "cold_litres": c, "switch_overs": s})
    print(f"AI margin {margin:+d} C: {e} kWh, {c} L cold, {s} switch-overs  "
          f"[{time.time() - t0:.0f}s]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "tradeoff_comparison.csv", index=False)

print("\n" + "=" * 78)
print("RESULTS (totals over the scenarios)")
print("=" * 78)
print(df.to_string(index=False))

zero = df[(df.strategy == "Thermostat") & (df.cold_litres <= 0.5)]
ai = df[df.strategy == "AI controller"]
print("\nHOW TO READ THIS")
if len(zero):
    best = zero.sort_values("energy_kwh").iloc[0]
    print(f"- Lowest-energy thermostat with no cold water: {best.setting} "
          f"({best.energy_kwh} kWh, {int(best.switch_overs)} switch-overs).")
if len(ai):
    least_cold = ai.sort_values("cold_litres").iloc[0]
    print(f"- AI setting with the least cold water: {least_cold.setting} "
          f"({least_cold.energy_kwh} kWh, {least_cold.cold_litres} L cold).")
print("- A strategy is better when it is lower AND to the left on the chart "
      "(less energy, less cold water).")
print("- Report what the numbers show, including if the thermostat wins.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color, marker in (("Thermostat", "#2563eb", "o"), ("AI controller", "#f97316", "s")):
        sub = df[df.strategy == name].sort_values("energy_kwh")
        ax.plot(sub.energy_kwh, sub.cold_litres, marker=marker, color=color, label=name)
        for _, r in sub.iterrows():
            ax.annotate(r.setting, (r.energy_kwh, r.cold_litres), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Backup energy used (kWh)")
    ax.set_ylabel("Hot water drawn below 45 C (litres)")
    ax.set_title("Energy versus comfort: thermostat and AI controller")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_comparison.png", dpi=150)
    print(f"\nSaved chart: {OUT_DIR / 'tradeoff_comparison.png'}")
except Exception:
    print("\n(matplotlib not available - chart skipped)")

print(f"Saved table: {OUT_DIR / 'tradeoff_comparison.csv'}")
print(f"Total time: {(time.time() - t0) / 60:.1f} min")