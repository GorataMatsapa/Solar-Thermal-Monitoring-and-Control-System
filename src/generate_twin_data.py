"""
STMaCS - GENERATE TRAINING DATA FROM THE DIGITAL TWIN

What this does
--------------
The digital twin is a calculator: give it weather and hot-water use and it works
out what the sensors would read. This script plays the twin forward for
simulated years (a few minutes of computer time), writes down what the sensors
read every 5 minutes, and saves it in the SAME file format as the system 455
dataset. The existing training scripts then run on it unchanged.

What goes in (all simulated, nothing measured)
----------------------------------------------
  Weather      : seasonal clear-sky curve for Gaborone, dimmed by a random
                 clear / partly cloudy / overcast sequence; seasonal ambient
                 temperature with daily swing; seasonal mains water temperature
  Hot-water use: discrete draw-off events (showers, taps). The time-of-day and
                 day-of-week pattern is copied from system 455's real data when
                 the 455 file is available; the daily volume is scaled to the
                 caravan (default 150 L/day), with day-to-day variation
  Heater       : thermostat with a random setpoint (40-56 C) on 20% of days,
                 and 1-3 forced-on runs (30-150 min window) on another 50% of days
                 (like a controller would do), so the data contains plenty of
                 heater events
  Pump         : normally AUTO; on about 10% of days a manual ON or OFF window
                 (so the demo's manual override is also covered)
  Sensor noise : small random noise (default 0.1 C) and 0.1 C rounding on the
                 temperature sensors, like real sensors

What comes out
--------------
  system455_60min_train.csv / system455_60min_test.csv  (same columns and
  names as the real data, so existing scripts run unchanged),
  twin_data_summary.txt

IMPORTANT: a model trained on this data learns the TWIN, so it will look very
accurate on twin data. That demonstrates the pipeline; it says nothing about
real-world accuracy. Real-world accuracy evidence is the system 455 result.

Usage (run from the project folder)
-----------------------------------
  python src\\generate_twin_data.py                  2 simulated years
  python src\\generate_twin_data.py --years 3
  python src\\generate_twin_data.py --quick          30 days, smoke test
  python src\\generate_twin_data.py --twin-path PATH\\digital_twin_model.py
"""

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--years", type=float, default=2.0)
parser.add_argument("--quick", action="store_true", help="30 simulated days")
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--twin-path", type=str, default=None,
                    help="path to digital_twin_model.py if it is not found automatically")
parser.add_argument("--profile-csv", type=str, default=None,
                    help="system 455 60-min CSV used for the hourly/weekday demand shape")
parser.add_argument("--daily-use-l", type=float, default=150.0)
parser.add_argument("--heater-w", type=float, default=5000.0)
parser.add_argument("--sensor-noise", type=float, default=0.1)
parser.add_argument("--start", type=str, default="2025-08-27")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
OUT_DIR = Path(args.out_dir) if args.out_dir else BASE_DIR / "data" / "processed" / "twin_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROFILE_CSV = BASE_DIR / "data" / "processed" / "ml" / "system455_60min_train.csv"

SAMPLE_S = 300                 # logged every 5 minutes (same as system 455)
DT_S = 60                      # twin steps at 1 minute, like the dashboard
SUB = SAMPLE_S // DT_S
SPINUP_DAYS = 7
TRAIN_FRACTION = 0.80
HEATER_THERMOSTAT_DAY_PROBABILITY = 0.20   # thermostat on all day, random setpoint
HEATER_WINDOW_DAY_PROBABILITY = 0.50       # 1-3 forced-on runs per day
MANUAL_PUMP_DAY_PROBABILITY = 0.10

rng = np.random.default_rng(args.seed)
summary_lines = []


def log(text=""):
    print(text)
    summary_lines.append(text)


# =============================================================================
# LOAD THE TWIN
# =============================================================================

def load_twin_module():
    candidates = []
    if args.twin_path:
        candidates.append(Path(args.twin_path))
    for base in (HERE, HERE.parent, HERE.parent / "src"):
        candidates.append(base / "utils" / "digital_twin_model.py")
        candidates.append(base / "digital_twin_model.py")
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("digital_twin_model", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError(
        "Could not find digital_twin_model.py. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPass its location with --twin-path.")


twin_module, twin_path = load_twin_module()
BaseTwin = twin_module.HOTSETDigitalTwin
PUMP_MODES = [twin_module.PUMP_AUTO, twin_module.PUMP_MANUAL_ON,
              twin_module.PUMP_MANUAL_OFF]


class DrivenTwin(BaseTwin):
    """The twin driven by an external irradiance value instead of its bell curve."""

    def __init__(self, **kwargs):
        self._irr_now = 0.0
        super().__init__(**kwargs)

    def calculate_solar_irradiance(self, simulation_time_seconds):
        return self._irr_now


# =============================================================================
# SCENARIO GENERATORS
# =============================================================================

# Fallback hourly shape (the predictive control page's default profile)
FALLBACK_HOURLY = np.array([
    0.5, 0.3, 0.2, 0.2, 0.4, 1.5, 4.0, 5.0, 3.0, 1.5, 1.0, 1.0,
    1.5, 1.0, 0.8, 0.8, 1.2, 3.0, 5.0, 5.0, 4.0, 2.5, 1.5, 0.8])


def load_demand_profile():
    """share[dow, hour] (rows sum to 1) and day_scale[dow] (mean 1)."""
    csv = Path(args.profile_csv) if args.profile_csv else DEFAULT_PROFILE_CSV
    if csv.exists():
        d = pd.read_csv(csv, usecols=["timestamp", "hotwater_flow"])
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        tbl = (d.groupby([d["timestamp"].dt.dayofweek, d["timestamp"].dt.hour])
                ["hotwater_flow"].mean().unstack(fill_value=0.0)
                .reindex(index=range(7), columns=range(24), fill_value=0.0)
                .to_numpy())
        row = tbl.sum(axis=1)
        if np.all(row > 0):
            return tbl / row[:, None], row / row.mean(), f"system 455 data ({csv.name})"
    share = np.tile(FALLBACK_HOURLY / FALLBACK_HOURLY.sum(), (7, 1))
    return share, np.ones(7), "built-in default profile"


def make_draw_profile(n_minutes, t_begin, share, day_scale, daily_use_l):
    """Hot-water draw in L/min for every minute, made of discrete events."""
    draw = np.zeros(n_minutes + 240)
    mean_volume = 25.0
    events_mean = daily_use_l / mean_volume
    for d in range(n_minutes // 1440 + 1):
        dow = (t_begin + pd.Timedelta(days=d)).dayofweek
        lam_day = events_mean * day_scale[dow] * rng.lognormal(-0.045, 0.3)
        for h in range(24):
            for _ in range(rng.poisson(lam_day * share[dow, h])):
                volume = float(np.clip(
                    rng.lognormal(math.log(mean_volume) - 0.125, 0.5), 8.0, 80.0))
                flow = rng.uniform(4.0, 9.0)                       # L/min
                minute = int(d * 1440 + h * 60 + rng.uniform(0, 60))
                remaining = volume
                while remaining > 1e-9 and minute < len(draw):
                    use = min(flow, remaining)
                    draw[minute] += use
                    remaining -= use
                    minute += 1
    return draw[:n_minutes]


def make_weather(n_minutes, t_begin):
    """Irradiance (W/m2), ambient (C) and mains water temperature (C) per minute."""
    minutes = np.arange(n_minutes)
    ts = pd.date_range(t_begin, periods=n_minutes, freq="min")
    doy = ts.dayofyear.to_numpy().astype(float)
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0

    ang = 2 * np.pi * (doy - 355) / 365.25               # 0 at the December solstice
    daylength = 12.1 + 1.6 * np.cos(ang)                 # about 10.5 h to 13.7 h
    g_max = 920 + 80 * np.cos(ang)
    x = (hour - (12 - daylength / 2)) / daylength
    clear = np.where((x > 0) & (x < 1), g_max * np.sin(np.pi * np.clip(x, 0, 1)) ** 1.15, 0.0)

    # daily weather state: 0 clear, 1 partly cloudy, 2 overcast (more cloud in summer)
    n_days = n_minutes // 1440 + 1
    transition = np.array([[0.75, 0.18, 0.07], [0.35, 0.40, 0.25], [0.25, 0.35, 0.40]])
    states = np.zeros(n_days, dtype=int)
    for d in range(1, n_days):
        day_doy = (t_begin + pd.Timedelta(days=d)).dayofyear
        summer = 0.5 + 0.5 * math.cos(2 * math.pi * (day_doy - 15) / 365.25)
        p = transition[states[d - 1]] * np.array([1.0, 1 + 0.6 * summer, 1 + 1.0 * summer])
        states[d] = rng.choice(3, p=p / p.sum())
    mean_f = np.array([0.97, 0.60, 0.20])[states]
    sd_f = np.array([0.03, 0.22, 0.08])[states]
    low_f = np.array([0.85, 0.12, 0.05])[states]

    n5 = n_minutes // 5 + 3
    ar = np.zeros(n5)
    eps = rng.standard_normal(n5)
    rho = 0.97
    for i in range(1, n5):
        ar[i] = rho * ar[i - 1] + math.sqrt(1 - rho ** 2) * eps[i]
    noise = np.interp(minutes / 5.0, np.arange(n5), ar)
    day_idx = minutes // 1440
    factor = np.clip(mean_f[day_idx] + sd_f[day_idx] * noise, low_f[day_idx], 1.0)
    irradiance = clear * factor

    anomaly = np.zeros(n_days)
    for d in range(1, n_days):
        anomaly[d] = 0.7 * anomaly[d - 1] + 2.5 * math.sqrt(1 - 0.49) * rng.standard_normal()
    anomaly_min = np.interp(minutes, (np.arange(n_days) + 0.5) * 1440, anomaly)
    season = np.cos(2 * np.pi * (doy - 15) / 365.25)      # +1 mid-January
    ambient = 19.5 + 6.5 * season + anomaly_min + 7.0 * np.cos(2 * np.pi * (hour - 15) / 24)
    cold = 22.0 + 4.0 * season
    return irradiance, ambient, cold, states


def make_schedules(n_minutes):
    """Per-minute heater setpoint (0 = heater disabled) and pump mode (0 auto, 1 on, 2 off)."""
    n_days = n_minutes // 1440 + 1
    heater_sp = np.zeros(n_days * 1440)
    pump_mode = np.zeros(n_days * 1440, dtype=int)
    for d in range(n_days):
        base = d * 1440
        r = rng.random()
        if r < HEATER_THERMOSTAT_DAY_PROBABILITY:
            heater_sp[base:base + 1440] = rng.integers(40, 57)       # thermostat all day
        elif r < HEATER_THERMOSTAT_DAY_PROBABILITY + HEATER_WINDOW_DAY_PROBABILITY:
            for _ in range(int(rng.integers(1, 4))):                 # forced-on runs, like
                start = base + int(rng.uniform(0, 22 * 60))          # a controller would do
                length = int(rng.uniform(30, 150))
                heater_sp[start:start + length] = rng.uniform(60, 85)
        if rng.random() < MANUAL_PUMP_DAY_PROBABILITY:
            start = base + int(rng.uniform(6 * 60, 16 * 60))
            length = int(rng.uniform(60, 240))
            pump_mode[start:start + length] = rng.choice([1, 2])
    return heater_sp[:n_minutes], pump_mode[:n_minutes]


# =============================================================================
# RUN THE TWIN
# =============================================================================

sim_days = 30 if args.quick else int(round(args.years * 365))
t_start = pd.Timestamp(args.start)
t_begin = t_start - pd.Timedelta(days=SPINUP_DAYS)
n_minutes = (sim_days + SPINUP_DAYS) * 1440
spin_minutes = SPINUP_DAYS * 1440

log("=" * 80)
log("DIGITAL-TWIN DATA GENERATION" + ("  [QUICK]" if args.quick else ""))
log("=" * 80)
log(f"Twin model : {twin_path}")
log(f"Simulating : {sim_days} days from {t_start.date()} "
    f"(+{SPINUP_DAYS} days spin-up, discarded)")
log(f"Heater     : {args.heater_w:.0f} W | thermostat days "
    f"{HEATER_THERMOSTAT_DAY_PROBABILITY:.0%}, forced-run days "
    f"{HEATER_WINDOW_DAY_PROBABILITY:.0%} | daily hot-water use "
    f"{args.daily_use_l:.0f} L")

share, day_scale, profile_source = load_demand_profile()
log(f"Demand shape: {profile_source}")

irr, amb, cold, weather_states = make_weather(n_minutes, t_begin)
draw = make_draw_profile(n_minutes, t_begin, share, day_scale, args.daily_use_l)
heater_sp, pump_modes = make_schedules(n_minutes)

twin = DrivenTwin(
    initial_tank_temperature=float(rng.uniform(25, 50)),
    ambient_temperature=float(amb[0]),
    cold_water_temperature=float(cold[0]),
    heater_power=0.0,
)

records = []
last_mode = 0
t_run = time.time()
for k in range(n_minutes):
    twin._irr_now = float(irr[k])
    twin.set_ambient_temperature(amb[k])
    twin.cold_water_temperature = float(cold[k])
    twin.set_hot_water_demand(draw[k] / 60.0)               # L/min -> L/s
    sp = heater_sp[k]
    twin.set_heater(args.heater_w if sp > 0 else 0.0, sp if sp > 0 else 55.0)
    if pump_modes[k] != last_mode:
        twin.set_pump_mode(PUMP_MODES[pump_modes[k]])
        last_mode = pump_modes[k]
    twin.step(DT_S)

    if k >= spin_minutes and (k + 1 - spin_minutes) % SUB == 0:
        hm = twin.heat_meter_readings()
        records.append((
            t_begin + pd.Timedelta(minutes=k + 1),
            twin.top_temperature, twin.middle_temperature, twin.bottom_temperature,
            twin.collector_temperature,
            twin.flow_rate * 60.0,                              # L/min -> L/h (as in 455)
            int(twin.pump_on),
            int(twin.heater_on and twin.heater_power > 0),
            twin.solar_irradiance, amb[k],
            hm["solar_loop"]["power"] / 1000.0,
            hm["solar_loop"]["supply_temperature"],
            hm["solar_loop"]["return_temperature"],
            float(np.mean(draw[k - SUB + 1:k + 1])) * 60.0,     # L/min -> L/h
            hm["domestic"]["power"] / 1000.0,
            twin.hot_water_temperature, twin.cold_water_temperature,
        ))
    if (k + 1) % (30 * 1440) == 0:
        log(f"  ... simulated {(k + 1) // 1440 - SPINUP_DAYS} days "
            f"({time.time() - t_run:.0f}s)")

log(f"Simulation finished in {time.time() - t_run:.0f}s "
    f"({len(records):,} five-minute samples).")

# =============================================================================
# BUILD THE DATAFRAME IN THE SYSTEM 455 FORMAT
# =============================================================================

cols = ["timestamp", "buffer_top", "buffer_centre", "buffer_bottom", "collector_temp",
        "solar_flow", "solar_pump", "electric_heater", "solar_radiation", "ambient",
        "solar_power", "solar_temp_hot", "solar_temp_cold", "hotwater_flow",
        "hotwater_power", "hotwater_temp_hot", "hotwater_temp_cold"]
df = pd.DataFrame(records, columns=cols)

# sensor noise and 0.1 C resolution on the temperature sensors
for c in ["buffer_top", "buffer_centre", "buffer_bottom", "collector_temp",
          "solar_temp_hot", "solar_temp_cold", "hotwater_temp_hot", "hotwater_temp_cold"]:
    df[c] = (df[c] + rng.normal(0.0, args.sensor_noise, len(df))).round(1)
df["solar_flow"] = df["solar_flow"].round(0)
df["hotwater_flow"] = df["hotwater_flow"].round(0)
df["solar_radiation"] = df["solar_radiation"].round(0)
df["solar_power"] = df["solar_power"].round(2)
df["hotwater_power"] = df["hotwater_power"].round(2)
df["pwm"] = df["solar_pump"] * 100

df["thermal_stratification_top_centre"] = df["buffer_top"] - df["buffer_centre"]
df["thermal_stratification_centre_bottom"] = df["buffer_centre"] - df["buffer_bottom"]
df["thermal_stratification_top_bottom"] = df["buffer_top"] - df["buffer_bottom"]
df["collector_minus_buffer_top"] = df["collector_temp"] - df["buffer_top"]
df["collector_minus_buffer_centre"] = df["collector_temp"] - df["buffer_centre"]
df["collector_minus_buffer_bottom"] = df["collector_temp"] - df["buffer_bottom"]
df["solar_loop_delta_t"] = df["solar_temp_hot"] - df["solar_temp_cold"]
df["hotwater_delta_t"] = df["hotwater_temp_hot"] - df["hotwater_temp_cold"]

ts = df["timestamp"]
frac_hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
df["hour"] = ts.dt.hour
df["day_of_week"] = ts.dt.dayofweek
df["day_of_year"] = ts.dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24.0)
df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

LAGS = (1, 3, 6, 12)
for ch in ["buffer_top", "buffer_centre", "buffer_bottom", "collector_temp",
           "solar_radiation", "solar_flow", "hotwater_flow"]:
    for k in LAGS:
        df[f"{ch}_lag_{k}"] = df[ch].shift(k)
for ch in ["buffer_top", "buffer_centre", "buffer_bottom", "collector_temp"]:
    for k in LAGS:
        df[f"{ch}_delta_{k}"] = df[ch] - df[f"{ch}_lag_{k}"]

df["target_buffer_top_60min"] = df["buffer_top"].shift(-12)

predictor_columns = (
    ["buffer_top", "buffer_centre", "buffer_bottom", "collector_temp",
     "solar_radiation", "solar_flow", "solar_power", "solar_temp_hot",
     "solar_temp_cold", "hotwater_flow", "hotwater_power", "hotwater_temp_hot",
     "hotwater_temp_cold", "pwm", "solar_pump", "electric_heater",
     "thermal_stratification_top_centre", "thermal_stratification_centre_bottom",
     "thermal_stratification_top_bottom", "collector_minus_buffer_top",
     "collector_minus_buffer_centre", "collector_minus_buffer_bottom",
     "solar_loop_delta_t", "hotwater_delta_t",
     "hour", "day_of_week", "day_of_year", "hour_sin", "hour_cos",
     "day_of_week_sin", "day_of_week_cos", "day_of_year_sin", "day_of_year_cos"]
    + [f"{ch}_lag_{k}" for ch in ["buffer_top", "buffer_centre", "buffer_bottom",
                                  "collector_temp", "solar_radiation", "solar_flow",
                                  "hotwater_flow"] for k in LAGS]
    + [f"{ch}_delta_{k}" for ch in ["buffer_top", "buffer_centre", "buffer_bottom",
                                    "collector_temp"] for k in LAGS]
)
assert len(predictor_columns) == 77, len(predictor_columns)

model_df = df[["timestamp"] + predictor_columns + ["target_buffer_top_60min"]]
model_df = model_df.dropna().reset_index(drop=True)

cut = int(len(model_df) * TRAIN_FRACTION)
train = model_df.iloc[:cut]
test = model_df.iloc[cut:]
train = train[train["timestamp"] + pd.Timedelta(minutes=60) < test["timestamp"].iloc[0]]

train.to_csv(OUT_DIR / "system455_60min_train.csv", index=False)
test.to_csv(OUT_DIR / "system455_60min_test.csv", index=False)

# =============================================================================
# SANITY SUMMARY
# =============================================================================

def cold_episodes(mask):
    changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    return len(starts), (np.median(ends - starts) * 5 / 60.0 if len(starts) else 0.0)


top = df["buffer_top"].to_numpy()
cold_mask = top < 45.0
n_ep, med_h = cold_episodes(cold_mask)
days_logged = len(df) * 5 / 1440.0
log("\n" + "-" * 80)
log("SANITY CHECK OF THE GENERATED DATA (twin vs real system 455)")
log("-" * 80)
log(f"Rows: train {len(train):,}  test {len(test):,}")
log(f"Train period: {train['timestamp'].iloc[0]}  ->  {train['timestamp'].iloc[-1]}")
log(f"Test period : {test['timestamp'].iloc[0]}  ->  {test['timestamp'].iloc[-1]}")
log(f"\n{'':<42}{'twin':>12}{'system 455':>14}")
log(f"{'Tank top (S3), median C':<42}{np.median(top):>12.1f}{'58.3':>14}")
log(f"{'Tank top (S3), 5th percentile C':<42}{np.percentile(top, 5):>12.1f}{'42.1':>14}")
log(f"{'Share of time S3 below 45 C':<42}{cold_mask.mean() * 100:>11.1f}%{'~10%':>14}")
log(f"{'Separate cold episodes (S3 < 45 C)':<42}{n_ep:>12}{'few':>14}")
log(f"{'Median cold episode length (hours)':<42}{med_h:>12.1f}{'':>14}")
log(f"{'Solar pump on':<42}{df['solar_pump'].mean() * 100:>11.1f}%{'43%':>14}")
log(f"{'Heater on':<42}{df['electric_heater'].mean() * 100:>11.1f}%{'4%':>14}")
log(f"{'Hot water being drawn (share of samples)':<42}"
    f"{(df['hotwater_flow'] > 0).mean() * 100:>11.1f}%{'55%':>14}")
log(f"{'Hot water per day, L':<42}"
    f"{df['hotwater_flow'].mean() * 24:>12.0f}{'~1640':>14}")
log(f"{'Solar energy to tank, kWh/day':<42}"
    f"{df['solar_power'].clip(lower=0).mean() * 24:>12.1f}{'~54':>14}")

log("\nShare of time S3 is below 45 C, by month (cold events must exist in the")
log("calibration period for the safety margins to be learnable):")
monthly = (df.assign(cold=cold_mask).groupby(df["timestamp"].dt.to_period("M"))["cold"].mean() * 100)
log("  " + "  ".join(f"{str(p)[2:]}:{v:.0f}%" for p, v in monthly.items()))

log("\nNext steps (run from the project folder):")
log(f"  python src\\train_final_models_v2.py --ml-dir {OUT_DIR} --out-dir models\\twin_v1")
log("  (add --quick first to check it runs)")
(OUT_DIR / "twin_data_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
log(f"\nSaved to: {OUT_DIR}")