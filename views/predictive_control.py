"""
Predictive Control page: HOTSET 300 L solar thermal monitoring & control.

What the page does
------------------
1. Reads the tank state (S1..S4) from the digital twin, live telemetry or manual entry.
2. Gets a weather forecast (synthetic for development, or Open-Meteo).
3. Forecasts the tank for the rest of today and tomorrow, with an uncertainty band.
4. Plans the cheapest backup-heater windows that keep S3 (hot water) above the
   comfort limit during the morning / evening demand windows.
5. Runs the controller: AUTO follows the plan; the user can force the heater ON or
   OFF for a while (manual override); a safety layer always has the last word.

The AI safety layer (trained forecaster + rule controller)
----------------------------------------------------------
Tick "Add the AI early-warning safety layer" in Settings. During the Digital Twin demo day
the trained models (utils/ai_safety_layer.py) watch the sensors every 5 simulated minutes
and can ADD heating when the pessimistic 1 h / 2 h forecast of S3 is at risk. Priority:
safety limit > manual override > AI safety net > the plan. See decide_command().

Where YOUR AI model plugs in (full-trajectory predictor, optional)
-------------------------------------------------------------------
The planner only talks to a `Predictor` (see below). The default `TwinPredictor`
is a physics baseline built on the digital twin, so everything on this page works
before the AI model exists. To use the trained model:

    * implement `AIPredictor.simulate()` so it returns the same DataFrame columns
      (see PREDICTOR_COLUMNS) for a given heater schedule and weather scenario, and
    * return an instance from `load_ai_predictor()`.

It then appears in the "Predictor" selector in the settings panel.

Where the REAL system plugs in
------------------------------
    * `read_live_sensors()`   -> S1..S4 from Firebase / ESP32
    * `send_heater_command()` -> the heater relay command

Both are stubs for now; commands are only logged on the page.
"""

from __future__ import annotations

import html as _html
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.digital_twin_model import HOTSETDigitalTwin
from utils.ai_closed_loop import render_tab as render_ai_tab
try:                                    # optional: trained forecaster as a safety layer
    from utils.ai_safety_layer import AISafetyLayer
except Exception:                       # the page keeps working without it
    AISafetyLayer = None
from utils import site_data as sd


# =========================================================
# CONSTANTS
# =========================================================
STEP_S = 300                 # simulation step of the forecast (s)
BLOCK_MIN = 15               # heater windows are planned in blocks of this length
MAX_BLOCKS = 16              # longest single heater window = 16 x 15 min = 4 h
LEAD_MIN = 10                # finish heating this long before a demand window starts
COMFORT_MARGIN_K = 1.0       # plan for S3 >= comfort limit + this margin
MIN_USEFUL_GAIN_K = 2.0      # when comfort is out of reach, only heat if S3 gains at least this
REPLAN_AFTER = timedelta(minutes=30)
CONTROL_TICK_S = 30          # the controller re-evaluates this often
MIN_CYCLE_MIN = 10           # heater min on-time / off-time (anti short-cycling)

SRC_TWIN = "Digital twin (simulation)"
SRC_LIVE = "Live telemetry"
SRC_MANUAL = "Manual entry"
SOURCES = [SRC_TWIN, SRC_LIVE, SRC_MANUAL]

WX_SYNTH = "Synthetic (development)"
WX_METEO = "Open-Meteo (needs latitude / longitude)"

PRED_TWIN = "Digital-twin physics baseline"
PRED_AI = "AI model"

# Columns every Predictor.simulate() must return (one row per STEP_S, first row = now).
PREDICTOR_COLUMNS = [
    "time", "s1", "s2", "s3", "s4",          # sensors, degC
    "heater_w", "solar_w", "draw_lpm",       # powers (W) and hot-water draw
    "irradiance", "ambient", "pump_on",
    "backup_kwh", "solar_kwh", "dhw_kwh",    # cumulative energies since the first row
]

# Share of the daily hot-water use drawn in each hour of the day (typical household).
_HOURLY_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.2, 0.4, 1.5,      # 00-05
    4.0, 5.0, 3.0, 1.5, 1.0, 1.0,      # 06-11
    1.5, 1.0, 0.8, 0.8, 1.2, 3.0,      # 12-17
    5.0, 5.0, 4.0, 2.5, 1.5, 0.8,      # 18-23
])
_HOURLY_SHARE = _HOURLY_WEIGHTS / _HOURLY_WEIGHTS.sum()


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass(frozen=True)
class Settings:
    source: str
    manual: tuple                # (s1, s2, s3, s4) used for manual entry / live fallback
    clock_offset_h: float
    weather_source: str
    lat: float
    lon: float
    cloud_today: float           # %
    cloud_tomorrow: float        # %
    peak_irradiance: float       # W/m2 at clear-sky noon (synthetic forecast)
    uncertainty: float           # 0..1, forecast error of the sun
    daily_use_l: float           # litres of tank water drawn per day
    morning: tuple               # (start hour, end hour) of the morning demand window
    evening: tuple
    comfort_min_c: float         # S3 must stay above this in the demand windows
    heater_power_w: float
    heater_target_c: float       # a planned heater run stops when S2 reaches this
    heater_inhibit_c: float      # heater is forced OFF above this S3 temperature
    thermostat_c: float          # setpoint of the plain-thermostat baseline
    price_per_kwh: float
    risk_tolerance: float        # 0..0.5 acceptable chance of missing comfort
    predictor: str
    use_ai: bool = False         # add the AI early-warning safety layer
    ai_model_dir: str = ""       # folder with config.json and the forecaster_*h.joblib files


@dataclass
class Snapshot:
    layers: np.ndarray
    s1: float
    s2: float
    s3: float
    s4: float
    origin: str
    note: str = ""


@dataclass
class Forecast:
    times: list
    irradiance: np.ndarray       # W/m2 on the collector
    ambient: np.ndarray          # degC
    source: str
    note: str = ""

    def seconds_from(self, start: datetime) -> np.ndarray:
        return np.array([(t - start).total_seconds() for t in self.times])


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    label: str = ""


@dataclass
class Plan:
    windows: list
    notes: list
    insufficient: list           # labels of demand windows that cannot be met


@dataclass(frozen=True)
class Scenario:
    irr_scale: float = 1.0
    demand_scale: float = 1.0
    label: str = "As forecast"


@dataclass
class Command:
    on: bool
    source: str                  # SAFETY / MANUAL / AI PLAN
    detail: str


# =========================================================
# TIME HELPERS
# =========================================================
def _now(cfg: Settings) -> datetime:
    """Local time; the offset slider lets you test the page at another time of day.
    While a Digital Twin "demo day" is running, the twin's virtual clock is used instead,
    so this page shows the plan, risk and decision log at the simulated time."""
    sim = st.session_state.get("sim_clock")
    if sim is not None and cfg.source == SRC_TWIN:
        return sim
    return datetime.now().replace(microsecond=0) + timedelta(hours=cfg.clock_offset_h)


def _floor(t: datetime, minutes: int) -> datetime:
    return t.replace(minute=(t.minute // minutes) * minutes, second=0, microsecond=0)


def _hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def _day_hm(t: datetime, ref: datetime) -> str:
    """HH:MM, with the weekday when it is not the same day as `ref`."""
    return _hm(t) if t.date() == ref.date() else t.strftime("%a %H:%M")


def demand_lps(cfg: Settings, t: datetime) -> float:
    """Hot-water draw (L/s) of the assumed daily profile at time t."""
    return cfg.daily_use_l * _HOURLY_SHARE[t.hour] / 3600.0


def comfort_windows(cfg: Settings, start: datetime, end: datetime) -> list:
    """Morning and evening demand windows that overlap [start, end]."""
    out = []
    day0 = datetime.combine(start.date(), dtime(0))
    for d in range(0, 4):
        base = day0 + timedelta(days=d)
        for label, (h0, h1) in (("Morning", cfg.morning), ("Evening", cfg.evening)):
            ws, we = base + timedelta(hours=h0), base + timedelta(hours=h1)
            if we > start and ws < end:
                out.append(Window(ws, we, label))
    return out


# =========================================================
# DATA ACQUISITION
# =========================================================
LIVE_MAX_AGE_S = 600         # live sensor values older than this are not trusted for planning


def read_live_sensors():
    """
    S1..S4 (degC) from Firebase site/latest, or None when live data cannot be trusted:
    nothing there, last reading older than LIVE_MAX_AGE_S, or any sensor invalid.
    """
    import time as _time
    raw, _ = sd.read_latest(sd.get_db())
    if not raw:
        return None
    d = sd.parse_record(raw)
    if d.get("epoch") is None or _time.time() - d["epoch"] > LIVE_MAX_AGE_S:
        return None
    if any(d.get(k) is None for k in ("s1", "s2", "s3", "s4")):
        return None
    return {k: d[k] for k in ("s1", "s2", "s3", "s4")}


def send_heater_command(on: bool, reason: str) -> None:
    """
    TODO: send the backup-heater command to the real system (Firebase / ESP32 relay).
    For now the command is only recorded in the decision log on this page.
    """
    return None


def _layers_from_sensors(s2: float, s3: float, s4: float, n: int) -> np.ndarray:
    """Spread the three tank sensors over the twin's layers (index 0 = top)."""
    return np.interp(np.arange(n), [0, (n - 1) // 2, n - 1], [s3, s2, s4])


def acquire_snapshot(cfg: Settings) -> Snapshot:
    """Current tank / collector state from the chosen source."""
    n = HOTSETDigitalTwin().tank_nodes
    s1, s2, s3, s4 = cfg.manual

    if cfg.source == SRC_TWIN:
        twin = st.session_state.get("twin_model")
        if twin is not None:
            return Snapshot(
                np.array(twin.node_temperatures, dtype=float),
                float(twin.collector_temperature), float(twin.middle_temperature),
                float(twin.top_temperature), float(twin.bottom_temperature),
                SRC_TWIN, "State of the model on the Digital Twin page.",
            )
        return Snapshot(
            np.full(n, 35.0), 20.0, 35.0, 35.0, 35.0, SRC_TWIN,
            "The Digital Twin page has not been opened yet: using a default 35 °C tank.",
        )

    note = ""
    origin = SRC_MANUAL
    if cfg.source == SRC_LIVE:
        live = read_live_sensors()
        if live:
            s1, s2, s3, s4 = (float(live[k]) for k in ("s1", "s2", "s3", "s4"))
            origin = SRC_LIVE
        else:
            note = ("Live telemetry unavailable (installation offline, last reading older than 10 min, or a sensor "
                    "invalid): using the manual sensor values.")
    return Snapshot(_layers_from_sensors(s2, s3, s4, n), s1, s2, s3, s4, origin, note)


# ---------------------------------------------------------
# Weather
# ---------------------------------------------------------
def synthetic_forecast(cfg: Settings, start: datetime, horizon_h: float) -> Forecast:
    """Clear-sky bell curve dimmed by the cloud sliders; a development stand-in."""
    t0 = start.replace(minute=0, second=0, microsecond=0)
    times = [t0 + timedelta(hours=h) for h in range(int(horizon_h) + 4)]
    irr, amb = [], []
    for t in times:
        h = t.hour + t.minute / 60.0
        clear = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
        cloud = (cfg.cloud_today if t.date() == start.date() else cfg.cloud_tomorrow) / 100.0
        irr.append(cfg.peak_irradiance * clear * (1 - 0.75 * cloud ** 3))
        amb.append(18.0 + 6.0 * math.cos(2 * math.pi * (h - 15) / 24.0))
    return Forecast(times, np.array(irr), np.array(amb), WX_SYNTH,
                    "Sliders, not a real forecast.")


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_open_meteo(lat: float, lon: float):
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,shortwave_radiation",
        "forecast_days": 3, "timezone": "auto",
    })
    with urllib.request.urlopen("https://api.open-meteo.com/v1/forecast?" + query, timeout=8) as r:
        hourly = json.load(r)["hourly"]
    return hourly["time"], hourly["temperature_2m"], hourly["shortwave_radiation"]


def open_meteo_forecast(cfg: Settings, start: datetime, horizon_h: float) -> Forecast:
    """
    Hourly forecast from Open-Meteo, in the location's local time.
    TODO: shortwave_radiation is irradiance on a horizontal plane; convert it to the
    collector plane (tilt / azimuth) for better accuracy.
    """
    stamps, temp, ghi = _fetch_open_meteo(round(cfg.lat, 3), round(cfg.lon, 3))
    times = [datetime.fromisoformat(s) for s in stamps]
    temp = pd.Series(temp, dtype=float).interpolate().bfill().ffill().to_numpy()
    ghi = pd.Series(ghi, dtype=float).interpolate().fillna(0.0).to_numpy()
    t0 = start.replace(minute=0, second=0, microsecond=0)
    keep = [i for i, t in enumerate(times)
            if t0 - timedelta(hours=1) <= t <= t0 + timedelta(hours=horizon_h + 3)]
    if len(keep) < horizon_h * 0.9:
        raise ValueError("forecast does not cover the planning horizon")
    return Forecast([times[i] for i in keep], ghi[keep], temp[keep], WX_METEO,
                    "Open-Meteo, horizontal irradiance used as collector irradiance.")


def get_forecast(cfg: Settings, start: datetime, horizon_h: float) -> Forecast:
    if cfg.weather_source == WX_METEO:
        if cfg.lat == 0.0 and cfg.lon == 0.0:
            fc = synthetic_forecast(cfg, start, horizon_h)
            fc.note = "Enter your latitude and longitude to use Open-Meteo. Using the synthetic forecast."
            return fc
        try:
            return open_meteo_forecast(cfg, start, horizon_h)
        except Exception as exc:                      # network, parsing, coverage...
            fc = synthetic_forecast(cfg, start, horizon_h)
            fc.note = f"Open-Meteo failed ({type(exc).__name__}). Using the synthetic forecast."
            return fc
    return synthetic_forecast(cfg, start, horizon_h)


# =========================================================
# PREDICTORS (physics baseline now, AI model later)
# =========================================================
class _ForecastTwin(HOTSETDigitalTwin):
    """The digital twin driven by an external irradiance series instead of its bell curve."""

    def __init__(self, t_s, irradiance, **kwargs):
        self._t_s = np.asarray(t_s, dtype=float)
        self._irr = np.asarray(irradiance, dtype=float)
        super().__init__(**kwargs)

    def calculate_solar_irradiance(self, simulation_time_seconds):
        return float(np.interp(simulation_time_seconds, self._t_s, self._irr))


def _heater_command(cfg: Settings, mode: str, windows: list, t: datetime, s3: float):
    """(power W, setpoint degC) the heater gets in a given step, for each strategy."""
    if mode == "none" or cfg.heater_power_w <= 0 or s3 >= cfg.heater_inhibit_c:
        return 0.0, cfg.heater_target_c
    if mode == "thermostat":
        return cfg.heater_power_w, cfg.thermostat_c
    in_window = any(w.start <= t < w.end for w in windows)       # mode == "plan"
    return (cfg.heater_power_w if in_window else 0.0), cfg.heater_target_c


class Predictor:
    """What the planner needs from a model. Subclass this for the AI model."""
    name = "base"

    def simulate(self, cfg: Settings, snap: Snapshot, fc: Forecast, start: datetime,
                 steps: int, windows: list, scenario: Scenario, heater_mode: str) -> pd.DataFrame:
        """
        Forecast `steps` steps of STEP_S seconds from `start`.
        heater_mode: "none" | "thermostat" (always-on thermostat at cfg.thermostat_c) |
                     "plan" (heater runs only inside `windows`, up to cfg.heater_target_c).
        Must return a DataFrame with PREDICTOR_COLUMNS, first row = the current state.
        """
        raise NotImplementedError


class TwinPredictor(Predictor):
    """Physics baseline: the HOTSET digital twin driven by the weather forecast."""
    name = PRED_TWIN

    def simulate(self, cfg, snap, fc, start, steps, windows, scenario, heater_mode):
        grid = np.arange(steps + 1) * STEP_S
        t_fc = fc.seconds_from(start)
        irr = np.interp(grid, t_fc, fc.irradiance) * scenario.irr_scale
        amb = np.interp(grid, t_fc, fc.ambient)
        times = [start + timedelta(seconds=int(s)) for s in grid]
        draw = np.array([demand_lps(cfg, t) for t in times]) * scenario.demand_scale

        m = _ForecastTwin(
            grid, irr,
            initial_tank_temperature=float(np.mean(snap.layers)),
            ambient_temperature=float(amb[0]),
            heater_power=0.0, heater_setpoint=cfg.heater_target_c,
        )
        m.node_temperatures = np.array(snap.layers, dtype=float).copy()
        m.collector_temperature = float(snap.s1)
        m.update_derived_state()

        backup_j = 0.0
        rows = []

        def record(k, heater_w):
            rows.append((
                times[k], m.collector_temperature, m.middle_temperature, m.top_temperature,
                m.bottom_temperature, heater_w, m.solar_power, m.hot_water_demand * 60.0,
                irr[k], amb[k], bool(m.pump_on), backup_j / 3.6e6,
                m.hm_solar_energy_j / 3.6e6, m.hm_domestic_energy_j / 3.6e6,
            ))

        record(0, 0.0)
        for k in range(1, steps + 1):
            m.set_ambient_temperature(amb[k])
            m.set_hot_water_demand(draw[k - 1])
            power, setpoint = _heater_command(cfg, heater_mode, windows, times[k - 1], m.top_temperature)
            m.set_heater(power, setpoint)
            m.step(STEP_S)
            heater_w = power if m.heater_on else 0.0
            backup_j += heater_w * STEP_S
            record(k, heater_w)

        df = pd.DataFrame(rows, columns=PREDICTOR_COLUMNS)
        df["time"] = pd.to_datetime(df["time"])
        return df


class AIPredictor(Predictor):
    """
    Skeleton for the trained model.

    A suitable model takes (current S1..S4, weather forecast, hot-water demand
    profile, heater schedule) and predicts the S1..S4 trajectories; backup energy
    follows from the heater schedule. Build the feature frame here, call your model
    and return a DataFrame with PREDICTOR_COLUMNS (same grid as TwinPredictor).
    """
    name = PRED_AI

    def __init__(self, model):
        self.model = model

    def simulate(self, cfg, snap, fc, start, steps, windows, scenario, heater_mode):
        raise NotImplementedError("Connect the trained model here.")


def load_ai_predictor():
    """
    TODO: load your trained model and return AIPredictor(model), e.g.

        import joblib
        return AIPredictor(joblib.load("models/backup_heater_model.joblib"))

    Return None while there is no model; the page then offers only the physics baseline.
    """
    return None


def available_predictors() -> dict:
    options = {PRED_TWIN: TwinPredictor()}
    ai = load_ai_predictor()
    if ai is not None:
        options[PRED_AI] = ai
    return options


# =========================================================
# PLANNER
# =========================================================
def ensemble_scenarios(uncertainty: float) -> list:
    """3 sun levels x 3 demand levels. Index 4 is the nominal (as forecast) member."""
    u = min(max(uncertainty, 0.0), 1.0)
    return [Scenario(max(0.0, i), d, f"sun {i:.2f} / use {d:.2f}")
            for i in (1.0 - u, 1.0, 1.0 + 0.3 * u) for d in (0.8, 1.0, 1.25)]


def planning_scenario(cfg: Settings) -> Scenario:
    """The weather the plan must survive: more cautious when risk tolerance is low."""
    w = min(max(cfg.risk_tolerance / 0.5, 0.0), 1.0)
    u = cfg.uncertainty
    return Scenario((1 - u) * (1 - w) + 1.0 * w, 1.25 * (1 - w) + 1.0 * w, "planning")


def min_s3(df: pd.DataFrame, w: Window) -> float:
    sel = df[(df["time"] >= w.start) & (df["time"] <= w.end)]
    return float(sel["s3"].min()) if len(sel) else float("inf")


def plan_backup(predictor: Predictor, cfg: Settings, snap: Snapshot, fc: Forecast,
                start: datetime, steps: int, comfort_wins: list) -> Plan:
    """
    For each demand window (in time order) find the SHORTEST heater run, finishing
    shortly before the window opens, that keeps S3 above the comfort limit even in a
    cautious weather scenario. Heating late wastes the least heat to standing losses.
    """
    windows, notes, insufficient = [], [], []
    if cfg.heater_power_w <= 0:
        return Plan([], ["No backup heater configured (power = 0 W)."], [])

    scenario = planning_scenario(cfg)
    block = timedelta(minutes=BLOCK_MIN)

    for w in comfort_wins:
        if w.end <= start:
            continue
        n_steps = min(steps, int(math.ceil((w.end - start).total_seconds() / STEP_S)))

        def lowest_s3(candidate: list) -> float:
            df = predictor.simulate(cfg, snap, fc, start, n_steps, candidate, scenario, "plan")
            return min_s3(df, w)

        def meets(candidate: list) -> bool:
            return lowest_s3(candidate) >= cfg.comfort_min_c + COMFORT_MARGIN_K

        if meets(windows):
            continue

        latest_end = w.start - timedelta(minutes=LEAD_MIN)
        if latest_end - start < block:            # too late to finish before the window opens
            max_n = min(MAX_BLOCKS, int((w.end - start) / block))

            def make(n, _s=start):
                return Window(_s, _s + n * block, f"{w.label} demand")
        else:
            max_n = min(MAX_BLOCKS, int((latest_end - start) / block))

            def make(n, _e=latest_end):
                return Window(_e - n * block, _e, f"{w.label} demand")

        if max_n < 1:
            insufficient.append(w.label)
            notes.append(f"{w.label} demand: no time left to heat.")
            continue
        best = lowest_s3(windows + [make(max_n)])
        if best < cfg.comfort_min_c + COMFORT_MARGIN_K:
            # Comfort cannot be reached. Heat anyway only if it clearly helps.
            insufficient.append(w.label)
            gain = best - lowest_s3(windows)
            if gain >= MIN_USEFUL_GAIN_K:
                windows.append(make(max_n))
                notes.append(f"{w.label} demand: best effort. {max_n * BLOCK_MIN} min of backup lifts S3 "
                             f"by {gain:.1f} K to {best:.0f} °C (limit {cfg.comfort_min_c:.0f} °C).")
            else:
                notes.append(f"{w.label} demand cannot be recovered in time: backup would lift S3 by only "
                             f"{gain:.1f} K, so none is planned.")
            continue
        lo, hi = 1, max_n                          # smallest number of blocks that works
        while lo < hi:
            mid = (lo + hi) // 2
            if meets(windows + [make(mid)]):
                hi = mid
            else:
                lo = mid + 1
        windows.append(make(lo))

    return Plan(windows, notes, insufficient)


# =========================================================
# BUNDLE: everything the page shows, computed in one go
# =========================================================
def _energy_between(df: pd.DataFrame, col: str, a: datetime, b: datetime) -> float:
    sec = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    t0 = df["time"].iloc[0]
    fa = (pd.Timestamp(a) - t0).total_seconds()
    fb = (pd.Timestamp(b) - t0).total_seconds()
    return float(np.interp(fb, sec, df[col]) - np.interp(fa, sec, df[col]))


def _per_day(df: pd.DataFrame, start: datetime, end: datetime) -> list:
    midnight = datetime.combine(start.date() + timedelta(days=1), dtime(0))
    spans = [("Today (rest)", start, min(midnight, end))]
    if end > midnight:
        spans.append(("Tomorrow", midnight, end))
    return [{
        "day": name,
        "solar": _energy_between(df, "solar_kwh", a, b),
        "backup": _energy_between(df, "backup_kwh", a, b),
        "dhw": _energy_between(df, "dhw_kwh", a, b),
    } for name, a, b in spans]


def strategy_metrics(df: pd.DataFrame, comfort_wins: list, cfg: Settings) -> dict:
    mins = [min_s3(df, w) for w in comfort_wins if w.end > df["time"].iloc[0]]
    mins = [m for m in mins if np.isfinite(m)]
    backup, solar = float(df["backup_kwh"].iloc[-1]), float(df["solar_kwh"].iloc[-1])
    return {
        "backup_kwh": backup,
        "cost": backup * cfg.price_per_kwh,
        "windows_below": int(sum(m < cfg.comfort_min_c for m in mins)),
        "windows_total": len(mins),
        "min_s3": min(mins) if mins else float("nan"),
        "solar_share": solar / (solar + backup) if (solar + backup) > 0 else 0.0,
    }


def build_bundle(predictor: Predictor, cfg: Settings, snap: Snapshot, fc: Forecast,
                 now: datetime) -> dict:
    start = _floor(now, 5)
    end = datetime.combine(start.date() + timedelta(days=2), dtime(0))   # end of tomorrow
    steps = int((end - start).total_seconds() // STEP_S)
    wins = comfort_windows(cfg, start, end)

    plan = plan_backup(predictor, cfg, snap, fc, start, steps, wins)

    def run(windows, scenario, mode):
        return predictor.simulate(cfg, snap, fc, start, steps, windows, scenario, mode)

    scenarios = ensemble_scenarios(cfg.uncertainty)
    plan_members = [run(plan.windows, s, "plan") for s in scenarios]
    none_members = [run([], s, "none") for s in scenarios]
    nominal = scenarios[4]
    plan_nom, none_nom = plan_members[4], none_members[4]
    thermo_nom = run([], nominal, "thermostat")

    s3_stack = np.vstack([m["s3"].to_numpy() for m in plan_members])
    risk = []
    for w in wins:
        if w.end <= start:
            continue
        risk.append({
            "window": w,
            "p_plan": float(np.mean([min_s3(m, w) < cfg.comfort_min_c for m in plan_members])),
            "p_none": float(np.mean([min_s3(m, w) < cfg.comfort_min_c for m in none_members])),
            "min_plan": min_s3(plan_nom, w),
            "min_none": min_s3(none_nom, w),
        })

    plan_rows = []
    for w in plan.windows:
        kwh = _energy_between(plan_nom, "backup_kwh", w.start, w.end)
        plan_rows.append({"window": w, "kwh": kwh, "cost": kwh * cfg.price_per_kwh})

    bundle = {
        "start": start, "end": end, "steps": steps, "snap": snap, "fc": fc,
        "plan": plan, "plan_rows": plan_rows, "comfort_wins": wins, "risk": risk,
        "plan_nominal": plan_nom, "none_nominal": none_nom, "thermo_nominal": thermo_nom,
        "band_lo": np.percentile(s3_stack, 10, axis=0),
        "band_hi": np.percentile(s3_stack, 90, axis=0),
        "days": _per_day(plan_nom, start, end),
        "m_plan": strategy_metrics(plan_nom, wins, cfg),
        "m_none": strategy_metrics(none_nom, wins, cfg),
        "m_thermo": strategy_metrics(thermo_nom, wins, cfg),
        "made_at": now, "predictor": predictor.name,
    }
    bundle["reasons"] = explain(bundle, cfg)
    return bundle


def explain(b: dict, cfg: Settings) -> list:
    """Plain-language reasons behind the plan, from the numbers above."""
    out = []
    d = b["days"]
    sun = f"{d[0]['solar']:.1f} kWh to the tank for the rest of today"
    if len(d) > 1:
        sun += f" and {d[1]['solar']:.1f} kWh tomorrow"
    out.append(f"Solar forecast ({b['fc'].source}): {sun}.")
    start = b["start"]
    for r in b["risk"]:
        w = r["window"]
        tag = f"{w.label} demand {_day_hm(w.start, start)}-{_hm(w.end)}"
        used = [row for row in b["plan_rows"] if row["window"].label == f"{w.label} demand"
                and row["window"].end <= w.start + timedelta(minutes=1)
                and row["window"].end >= w.start - timedelta(hours=1)]
        if r["min_none"] >= cfg.comfort_min_c + COMFORT_MARGIN_K and r["p_none"] <= cfg.risk_tolerance:
            out.append(f"{tag}: S3 stays at {r['min_none']:.0f} °C or more without backup, so none is needed.")
        elif used:
            row = used[0]
            out.append(f"{tag}: without backup S3 drops to {r['min_none']:.0f} °C (limit {cfg.comfort_min_c:.0f} °C, "
                       f"{r['p_none']:.0%} risk). Backup {_day_hm(row['window'].start, start)}-{_hm(row['window'].end)} "
                       f"(about {row['kwh']:.1f} kWh) brings it to {r['min_plan']:.0f} °C.")
        else:
            out.append(f"{tag}: S3 drops to {r['min_none']:.0f} °C without backup ({r['p_none']:.0%} risk).")
    out.extend(b["plan"].notes)
    return out


# =========================================================
# CONTROLLER
# =========================================================
def decide_command(now: datetime, plan: Plan, override, snap: Snapshot, cfg: Settings,
                   last_on, last_change, ai=None) -> Command:
    """
    Priority: 1. safety   2. manual override   3. AI safety net   4. plan
    (+ anti short-cycling on 3 and 4). `ai` is an AIOpinion from utils/ai_safety_layer.py
    or None. The safety net can only ADD heating the plan did not schedule.
    """
    if cfg.heater_power_w <= 0:
        return Command(False, "SETTINGS", "no backup heater configured")
    if snap.s3 >= cfg.heater_inhibit_c:
        return Command(False, "SAFETY", f"S3 {snap.s3:.0f} °C is above the {cfg.heater_inhibit_c:.0f} °C heater limit")

    if override is not None and now < override["until"]:
        word = "ON" if override["on"] else "OFF"
        return Command(override["on"], "MANUAL", f"forced {word} until {_hm(override['until'])}")

    active = next((w for w in plan.windows if w.start <= now < w.end), None)
    if active is not None and snap.s2 < cfg.heater_target_c:
        wanted = Command(True, "AI PLAN", f"planned run {_hm(active.start)}-{_hm(active.end)}")
    elif active is not None:
        wanted = Command(False, "AI PLAN", f"S2 reached {cfg.heater_target_c:.0f} °C")
    else:
        wanted = Command(False, "AI PLAN", "no run planned right now")

    if ai is not None and getattr(ai, "ready", False) and ai.want_on and not wanted.on:
        wanted = Command(True, "AI SAFETY NET", f"forecast risk ({ai.reason})")

    if last_on is not None and wanted.on != last_on and last_change is not None \
            and (now - last_change) < timedelta(minutes=MIN_CYCLE_MIN):
        return Command(last_on, "AI PLAN", f"holding for the {MIN_CYCLE_MIN} min minimum on/off time")
    return wanted


def _log(now: datetime, event: str, source: str, detail: str) -> None:
    log = st.session_state.setdefault("pc_log", [])
    log.insert(0, {"time": now.strftime("%a %H:%M:%S"), "event": event, "source": source, "detail": detail})
    del log[200:]


def _apply_command(cmd: Command, now: datetime) -> None:
    previous = st.session_state.get("pc_cmd_on")
    if previous is None:
        st.session_state.pc_cmd_on = cmd.on
        st.session_state.pc_last_change = now
        _log(now, f"Controller started, heater {'ON' if cmd.on else 'OFF'}", cmd.source, cmd.detail)
    elif previous != cmd.on:
        send_heater_command(cmd.on, f"{cmd.source}: {cmd.detail}")
        st.session_state.pc_cmd_on = cmd.on
        st.session_state.pc_last_change = now
        _log(now, f"Heater {'ON' if cmd.on else 'OFF'}", cmd.source, cmd.detail)


def _active_override(now: datetime):
    ov = st.session_state.get("pc_override")
    if ov is not None and now >= ov["until"]:
        _log(now, "Manual override ended", "MANUAL", "back to AUTO")
        st.session_state.pc_override = None
        return None
    return ov


def _set_override(on: bool) -> None:
    cfg = st.session_state.get("pc_cfg")
    if cfg is None:
        return
    now = _now(cfg)
    minutes = int(st.session_state.get("pc_override_minutes", 60))
    st.session_state.pc_override = {"on": on, "until": now + timedelta(minutes=minutes)}
    _log(now, f"Manual override: force {'ON' if on else 'OFF'} for {minutes} min", "MANUAL", "operator action")


def _clear_override() -> None:
    cfg = st.session_state.get("pc_cfg")
    if cfg is not None and st.session_state.get("pc_override") is not None:
        _log(_now(cfg), "Manual override cancelled", "MANUAL", "back to AUTO")
    st.session_state.pc_override = None


# =========================================================
# BRIDGE TO THE DIGITAL TWIN PAGE (SIMULATION "demo day")
# =========================================================
SIM_REPLAN_AFTER = timedelta(minutes=60)   # simulated minutes between replans (a replan takes a few seconds)


def actual_weather(cfg: Settings, t: datetime, cloud_pct: float, ambient_offset_c: float = 0.0):
    """The weather the twin ACTUALLY experiences (irradiance W/m2, ambient degC).
    Same shape as synthetic_forecast, but driven by its own cloud value, so the
    forecast the planner used can be wrong."""
    h = t.hour + t.minute / 60.0
    clear = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
    irr = cfg.peak_irradiance * clear * (1 - 0.75 * (cloud_pct / 100.0) ** 3)
    amb = 18.0 + 6.0 * math.cos(2 * math.pi * (h - 15) / 24.0) + ambient_offset_c
    return irr, amb


def _get_ai_layer(cfg: Settings):
    """Load the trained models once per run; never raise (the planner must keep working)."""
    ss = st.session_state
    if not cfg.use_ai:
        return None
    key = (cfg.ai_model_dir, round(cfg.heater_power_w))
    if ss.get("pc_ai_key") == key:
        return ss.get("pc_ai_layer")            # None if loading failed for this key
    ss.pc_ai_key = key
    ss.pc_ai_layer = None
    if AISafetyLayer is None:
        ss.pc_ai_status = {"ready": False, "want_on": False, "mode": "not loaded",
                           "reason": "utils/ai_safety_layer.py could not be imported", "forecast": {}}
        return None
    try:
        ss.pc_ai_layer = AISafetyLayer(cfg.ai_model_dir, cfg.heater_power_w)
    except Exception as exc:
        ss.pc_ai_status = {"ready": False, "want_on": False, "mode": "not loaded",
                           "reason": f"{type(exc).__name__}: {exc}", "forecast": {}}
    return ss.pc_ai_layer


def controller_step_for_twin(twin, sim_now: datetime):
    """
    One control decision for the simulated twin at virtual time `sim_now`.
    Called by the Digital Twin page every simulated minute (Streamlit only runs the page
    that is on screen, so the controller has to be driven from the twin page).
    Returns a Command, or None if this page's settings have not been created yet.
    """
    cfg = st.session_state.get("pc_cfg")
    if cfg is None:
        return None
    snap = Snapshot(
        np.array(twin.node_temperatures, dtype=float), float(twin.collector_temperature),
        float(twin.middle_temperature), float(twin.top_temperature), float(twin.bottom_temperature),
        SRC_TWIN, "State of the model on the Digital Twin page.",
    )
    bundle = st.session_state.get("pc_bundle")
    if bundle is None or not (timedelta(0) <= sim_now - bundle["made_at"] < SIM_REPLAN_AFTER):
        start = _floor(sim_now, 5)
        horizon_h = (datetime.combine(start.date() + timedelta(days=2), dtime(0)) - start).total_seconds() / 3600
        predictor = available_predictors().get(cfg.predictor, TwinPredictor())
        bundle = build_bundle(predictor, cfg, snap, get_forecast(cfg, start, horizon_h), sim_now)
        st.session_state.pc_bundle = bundle
        st.session_state.pc_sig = _signature(cfg, snap, sim_now)
        st.session_state.active_heater_plan = bundle["plan"].windows
        spans = ", ".join(f"{_hm(w.start)}-{_hm(w.end)}" for w in bundle["plan"].windows) or "none"
        _log(sim_now, "Replanned", "AI PLAN", f"heater windows: {spans}")
    ai = None
    layer = _get_ai_layer(cfg)
    if layer is not None:
        ai = layer.step(sim_now, snap.s1, snap.s2, snap.s3, snap.s4,
                        float(twin.flow_rate) * 60.0, bool(twin.pump_on))   # flow in L/h, as trained
        st.session_state.pc_ai_status = ai.as_dict()
    cmd = decide_command(sim_now, bundle["plan"], _active_override(sim_now), snap, cfg,
                         st.session_state.get("pc_cmd_on"), st.session_state.get("pc_last_change"), ai)
    _apply_command(cmd, sim_now)
    return cmd


# =========================================================
# SVG CHARTS (dark card, same look as the digital twin page)
# =========================================================
_FONT = 'font-family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"'


def _show_html(markup: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(markup.strip(), height=height)
    else:
        components.html(markup, height=height, scrolling=False)


def forecast_svg(b: dict, cfg: Settings, now: datetime) -> str:
    W, H = 1000, 410
    L, R, T, B = 58, 20, 54, 46
    plan_df, none_df = b["plan_nominal"], b["none_nominal"]
    t0 = plan_df["time"].iloc[0].to_pydatetime()
    t1 = plan_df["time"].iloc[-1].to_pydatetime()
    span = (t1 - t0).total_seconds()

    def X(t):
        return L + (t - t0).total_seconds() / span * (W - L - R)

    allv = np.concatenate([plan_df["s3"], none_df["s3"], plan_df["s4"],
                           b["band_lo"], b["band_hi"], [cfg.comfort_min_c]])
    ymin = 5 * math.floor((allv.min() - 2) / 5)
    ymax = 5 * math.ceil((allv.max() + 2) / 5)

    def Y(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - B)

    plot_h = H - T - B
    sel = slice(None, None, 3)                                   # 15-minute points
    times = [x.to_pydatetime() for x in plan_df["time"]]
    xs = [X(t) for t in times][sel]

    def line(values, color, width=2.4, dash=""):
        pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in zip(xs, np.asarray(values)[sel]))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    parts = []
    step = 5 if (ymax - ymin) <= 40 else 10
    for v in range(int(ymin), int(ymax) + 1, step):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#1e293b"/>')
        parts.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{v}</text>')
    parts.append(f'<text x="14" y="{T + plot_h / 2:.1f}" fill="#94a3b8" font-size="11" text-anchor="middle" '
                 f'transform="rotate(-90 14 {T + plot_h / 2:.1f})">Temperature (°C)</text>')

    first_midnight = datetime.combine(t0.date() + timedelta(days=1), dtime(0))
    if X(first_midnight) - L > 90:                       # label the day the chart starts on
        parts.append(f'<text x="{L + 4}" y="{H - B + 30}" fill="#cbd5e1" font-size="11" '
                     f'font-weight="700">{t0.strftime("%a %d %b")}</text>')
    tick = t0.replace(minute=0, second=0, microsecond=0)
    while tick < t0 or tick.hour % 6:
        tick += timedelta(hours=1)
    while tick <= t1:
        x = X(tick)
        if tick.hour == 0:
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" stroke="#334155"/>')
            if x < W - R - 70:                           # no room for a label at the right edge
                parts.append(f'<text x="{x + 4:.1f}" y="{H - B + 30}" fill="#cbd5e1" font-size="11" '
                             f'font-weight="700">{tick.strftime("%a %d %b")}</text>')
        else:
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" stroke="#1e293b"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - B + 16}" fill="#94a3b8" font-size="11" '
                     f'text-anchor="middle">{tick.strftime("%H:%M")}</text>')
        tick += timedelta(hours=6)

    for w in b["comfort_wins"]:
        xa, xb = X(max(w.start, t0)), X(min(w.end, t1))
        if xb > xa:
            parts.append(f'<rect x="{xa:.1f}" y="{T}" width="{xb - xa:.1f}" height="{plot_h}" fill="#38bdf8" opacity="0.10"/>')
            parts.append(f'<text x="{xa + 4:.1f}" y="{H - B - 6}" fill="#7dd3fc" font-size="10">{w.label} demand</text>')
    for w in b["plan"].windows:
        xa, xb = X(max(w.start, t0)), X(min(w.end, t1))
        parts.append(f'<rect x="{xa:.1f}" y="{T}" width="{max(xb - xa, 2):.1f}" height="{plot_h}" fill="#f97316" opacity="0.32"/>')
        parts.append(f'<text x="{xa + 4:.1f}" y="{T + 13}" fill="#fdba74" font-size="10" font-weight="700">'
                     f'Backup {_hm(w.start)}-{_hm(w.end)}</text>')

    band = [(x, v) for x, v in zip(xs, b["band_hi"][sel])] + \
           [(x, v) for x, v in reversed(list(zip(xs, b["band_lo"][sel])))]
    parts.append('<polygon points="' + " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in band) +
                 '" fill="#fb923c" opacity="0.20"/>')

    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(cfg.comfort_min_c):.1f}" y2="{Y(cfg.comfort_min_c):.1f}" '
                 f'stroke="#34d399" stroke-dasharray="6 5" stroke-width="1.5"/>')
    parts.append(f'<text x="{W - R - 4}" y="{Y(cfg.comfort_min_c) - 5:.1f}" fill="#34d399" font-size="10" '
                 f'text-anchor="end">comfort limit {cfg.comfort_min_c:.0f} °C</text>')

    parts.append(line(plan_df["s4"], "#38bdf8", 1.8))
    parts.append(line(none_df["s3"], "#94a3b8", 2.0, "6 5"))
    parts.append(line(plan_df["s3"], "#fb923c", 2.8))

    if t0 <= now <= t1:
        xn = X(now)
        parts.append(f'<line x1="{xn:.1f}" x2="{xn:.1f}" y1="{T}" y2="{H - B}" stroke="#e2e8f0" '
                     f'stroke-dasharray="3 4" stroke-opacity="0.8"/>')
        parts.append(f'<text x="{xn + 4:.1f}" y="{T - 6}" fill="#e2e8f0" font-size="10" font-weight="700">now</text>')

    for i, t in enumerate(times):                                  # hover read-outs, hourly
        if t.minute == 0:
            r = plan_df.iloc[i]
            tip = (f"{t.strftime('%a %H:%M')} · S3 {r['s3']:.1f} °C with plan, "
                   f"{none_df.iloc[i]['s3']:.1f} °C without backup · S4 {r['s4']:.1f} °C · "
                   f"heater {r['heater_w']:.0f} W")
            parts.append(f'<circle cx="{X(t):.1f}" cy="{Y(r["s3"]):.1f}" r="7" fill="#ffffff" fill-opacity="0">'
                         f'<title>{_html.escape(tip)}</title></circle>')

    legend = [("#fb923c", "S3 top, with plan"), ("#94a3b8", "S3 top, no backup"),
              ("#38bdf8", "S4 bottom, with plan"), ("#fb923c", "10-90 % range (S3)"),
              ("#f97316", "backup heater ON"), ("#38bdf8", "demand window")]
    lx, items = L, []
    for k, (c, text) in enumerate(legend):
        if k == 1:
            items.append(f'<line x1="{lx}" x2="{lx + 18}" y1="22" y2="22" stroke="{c}" stroke-width="2" stroke-dasharray="5 3"/>')
        elif k >= 3:
            items.append(f'<rect x="{lx}" y="16" width="18" height="11" fill="{c}" opacity="{0.45 if k == 4 else 0.3}"/>')
        else:
            items.append(f'<line x1="{lx}" x2="{lx + 18}" y1="22" y2="22" stroke="{c}" stroke-width="2.6"/>')
        items.append(f'<text x="{lx + 24}" y="26" fill="#cbd5e1" font-size="11">{text}</text>')
        lx += 24 + 7 * len(text) + 16

    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" {_FONT} '
            f'role="img" aria-label="Tank temperature forecast">'
            f'<rect width="{W}" height="{H}" rx="14" fill="#0b1220"/>' + "".join(items) + "".join(parts) + '</svg>')


def energy_svg(days: list) -> str:
    W, H = 520, 300
    L, R, T, B = 48, 16, 40, 44
    series = [("solar", "Solar to tank", "#fbbf24"), ("backup", "Backup heater", "#f97316"),
              ("dhw", "Hot water drawn", "#38bdf8")]
    vmax = max(max(d[k] for k, _, _ in series) for d in days) or 1.0
    top = math.ceil(vmax / 2) * 2 or 2
    plot_h = H - T - B

    def Y(v):
        return T + (top - v) / top * plot_h

    parts = []
    for v in range(0, top + 1, max(1, top // 4)):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#1e293b"/>')
        parts.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{v}</text>')
    parts.append(f'<text x="12" y="{T + plot_h / 2:.1f}" fill="#94a3b8" font-size="11" text-anchor="middle" '
                 f'transform="rotate(-90 12 {T + plot_h / 2:.1f})">kWh</text>')
    group_w = (W - L - R) / len(days)
    bar_w = group_w / (len(series) + 1.4)
    for gi, d in enumerate(days):
        gx = L + gi * group_w + bar_w * 0.7
        for si, (key, _, color) in enumerate(series):
            x, v = gx + si * bar_w, d[key]
            parts.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bar_w - 4:.1f}" height="{H - B - Y(v):.1f}" '
                         f'rx="3" fill="{color}"><title>{d["day"]}: {v:.2f} kWh</title></rect>')
            parts.append(f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{Y(v) - 5:.1f}" fill="#e2e8f0" font-size="10" '
                         f'text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="{L + gi * group_w + group_w / 2:.1f}" y="{H - B + 18}" fill="#cbd5e1" font-size="12" '
                     f'font-weight="700" text-anchor="middle">{d["day"]}</text>')
    lx = L
    for _, text, color in series:
        parts.append(f'<rect x="{lx}" y="14" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18}" y="24" fill="#cbd5e1" font-size="11">{text}</text>')
        lx += 18 + 7 * len(text) + 14
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" {_FONT} role="img" '
            f'aria-label="Energy balance per day"><rect width="{W}" height="{H}" rx="14" fill="#0b1220"/>'
            + "".join(parts) + '</svg>')


# =========================================================
# UI
# =========================================================
def _default_ai_dir() -> str:
    try:
        from utils.ai_closed_loop import _default_model_dir
        return str(_default_model_dir())
    except Exception:
        return "models/twin_v1"


def _ai_status_panel() -> None:
    """What the AI safety layer is thinking (only shown when it is switched on)."""
    cfg = st.session_state.get("pc_cfg")
    if cfg is None or not cfg.use_ai:
        return
    s = st.session_state.get("pc_ai_status")
    with st.container(border=True):
        st.markdown("**🤖 AI early-warning safety layer**")
        if not s:
            st.caption("Waiting for the Digital Twin demo day to start (tick 'Predictive demo day' there).")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", s["mode"])
        c2.metric("Asks for heater", "ON" if s.get("want_on") else "OFF")
        fc = s.get("forecast") or {}
        if fc:
            c3.metric("Pessimistic S3 in 1 h", f"{fc['1 h'][0]:.1f} °C",
                      delta=f"warn below {fc['1 h'][1]:.1f} °C", delta_color="off")
            st.caption(f"In 2 h: {fc['2 h'][0]:.1f} °C (warn below {fc['2 h'][1]:.1f} °C). {s['reason']}")
        else:
            st.caption(s["reason"])


def _settings_panel() -> Settings:
    predictors = available_predictors()
    with st.expander("⚙️ Settings: inputs, weather, comfort and heater", expanded=False):
        a, b, c = st.columns(3)

        with a:
            st.markdown("**Tank state and clock**")
            source = st.selectbox("Tank state from", SOURCES, key="pc_source")
            st.caption("Manual sensor values (used for Manual entry, and until live telemetry is connected)")
            m1, m2 = st.columns(2)
            with m1:
                s1 = st.number_input("S1 collector (°C)", -10.0, 150.0, 25.0, 0.5, key="pc_s1")
                s3 = st.number_input("S3 tank top (°C)", 0.0, 100.0, 42.0, 0.5, key="pc_s3")
            with m2:
                s2 = st.number_input("S2 tank middle (°C)", 0.0, 100.0, 38.0, 0.5, key="pc_s2")
                s4 = st.number_input("S4 tank bottom (°C)", 0.0, 100.0, 30.0, 0.5, key="pc_s4")
            clock = st.slider("Shift the clock (hours, for testing)", -12.0, 12.0, 0.0, 0.5, key="pc_clock")
            predictor = st.selectbox("Predictor", list(predictors), key="pc_predictor")
            use_ai = st.checkbox("Add the AI early-warning safety layer", value=False, key="pc_use_ai",
                                 help="Trained forecaster + rule controller. Used in the Digital Twin demo "
                                      "day: it can add heating when the 1 h / 2 h forecast of S3 is at risk.")
            ai_dir = st.text_input("Trained model folder", _default_ai_dir(), key="pc_ai_dir",
                                   disabled=not use_ai)
            if PRED_AI not in predictors:
                st.caption("AI model: not connected yet (see load_ai_predictor in predictive_control.py).")

        with b:
            st.markdown("**Weather**")
            wx = st.selectbox("Forecast source", [WX_SYNTH, WX_METEO], key="pc_wx")
            lat = st.number_input("Latitude", -90.0, 90.0, -24.6581, format="%.4f", key="pc_lat")
            lon = st.number_input("Longitude", -180.0, 180.0, 25.9088, format="%.4f", key="pc_lon")
            cloud_today = st.slider("Cloud cover today (%)", 0, 100, 20, key="pc_cloud0")
            cloud_tomorrow = st.slider("Cloud cover tomorrow (%)", 0, 100, 40, key="pc_cloud1")
            peak = st.slider("Clear-sky noon irradiance (W/m²)", 400, 1100, 900, 25, key="pc_peak")
            unc = st.slider("Forecast uncertainty of the sun (±%)", 0, 60, 30, key="pc_unc")

        with c:
            st.markdown("**Hot water, comfort and heater**")
            use = st.slider("Hot water drawn per day (L of tank water)", 50, 400, 150, 10, key="pc_use")
            morning = st.slider("Morning demand window (h)", 0, 12, (6, 9), key="pc_morning")
            evening = st.slider("Evening demand window (h)", 12, 24, (18, 21), key="pc_evening")
            comfort = st.slider("Comfort limit for S3 (°C)", 35, 60, 45, key="pc_comfort")
            power = st.slider("Backup heater power (W, 0 = none)", 0, 3000, 2000, 250, key="pc_power")
            target = st.slider("Heater stops when S2 reaches (°C)", 45, 75, 55, key="pc_target")
            inhibit = st.slider("Heater safety cut-off, S3 (°C)", 60, 90, 80, key="pc_inhibit")
            thermo = st.slider("Plain-thermostat baseline setpoint (°C)", 40, 65, 50, key="pc_thermo")
            price = st.number_input("Electricity price (per kWh)", 0.0, 100.0, 2.0, 0.1, key="pc_price")
            tol = st.slider("Acceptable chance of missing comfort (%)", 0, 40, 10, key="pc_tol")

    return Settings(
        source=source, manual=(s1, s2, s3, s4), clock_offset_h=clock, weather_source=wx,
        lat=lat, lon=lon, cloud_today=cloud_today, cloud_tomorrow=cloud_tomorrow,
        peak_irradiance=peak, uncertainty=unc / 100.0, daily_use_l=use, morning=morning,
        evening=evening, comfort_min_c=float(comfort), heater_power_w=float(power),
        heater_target_c=float(target), heater_inhibit_c=float(inhibit),
        thermostat_c=float(thermo), price_per_kwh=price, risk_tolerance=tol / 100.0,
        predictor=predictor, use_ai=bool(use_ai), ai_model_dir=str(ai_dir),
    )


def _signature(cfg: Settings, snap: Snapshot, now: datetime) -> str:
    return "|".join([
        repr(cfg), _floor(now, 15).isoformat(), repr(np.round(snap.layers, 0).tolist()),
        f"{round(snap.s1):d}",
    ])


def _controller_panel() -> None:
    """The decision card and manual override. Runs every few seconds as a fragment."""
    cfg = st.session_state.get("pc_cfg")
    bundle = st.session_state.get("pc_bundle")
    if cfg is None or bundle is None:
        return
    now = _now(cfg)
    snap = acquire_snapshot(cfg)
    override = _active_override(now)
    cmd = decide_command(now, bundle["plan"], override, snap, cfg,
                         st.session_state.get("pc_cmd_on"), st.session_state.get("pc_last_change"))
    _apply_command(cmd, now)

    upcoming = [w for w in bundle["plan"].windows if w.end > now]
    if cmd.on:
        active = next((w for w in bundle["plan"].windows if w.start <= now < w.end), None)
        headline = "Backup heater: ON"
        if cmd.source == "MANUAL":
            sub = cmd.detail
        elif active is not None:
            sub = f"running until {_hm(active.end)} or until S2 reaches {cfg.heater_target_c:.0f} °C"
        else:
            sub = cmd.detail
    elif upcoming:
        nxt = min(upcoming, key=lambda w: w.start)
        row = next((r for r in bundle["plan_rows"] if r["window"] == nxt), None)
        energy = f" · about {row['kwh']:.1f} kWh" if row else ""
        headline = "Backup heater: OFF"
        sub = f"next planned run {_day_hm(nxt.start, now)}-{_hm(nxt.end)}{energy}"
    else:
        headline = "Backup heater: OFF"
        sub = "no backup needed in the forecast"

    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            colour = "green" if cmd.on else "gray"
            st.markdown(f"### :{colour}[{headline}]")
            st.markdown(sub)
            st.caption(f"Decided by **{cmd.source}** · {cmd.detail} · checked {now.strftime('%H:%M:%S')}")
            for reason in bundle["reasons"][:3]:
                st.caption("• " + reason)
        with right:
            if override is not None:
                word = "ON" if override["on"] else "OFF"
                st.warning(f"MANUAL OVERRIDE: heater forced {word} until {_hm(override['until'])}")
            else:
                st.success("AUTO: following the plan")
            st.selectbox("Override for", [15, 30, 60, 120, 240], index=2, key="pc_override_minutes",
                         format_func=lambda m: f"{m} min" if m < 60 else f"{m // 60} h")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.button("⚡ Force ON", on_click=_set_override, args=(True,), key="pc_force_on",
                          use_container_width=True)
            with b2:
                st.button("⛔ Force OFF", on_click=_set_override, args=(False,), key="pc_force_off",
                          use_container_width=True)
            with b3:
                st.button("↩ Auto", on_click=_clear_override, key="pc_auto", use_container_width=True)
            st.caption("Safety limits always apply, even in manual override. "
                       "Commands are logged only until send_heater_command() is connected.")

    if now - bundle["made_at"] > REPLAN_AFTER:
        st.rerun()


def _whatif_table(b: dict, cfg: Settings, predictor: Predictor, choice: str) -> pd.DataFrame:
    scen = {
        "As forecast": Scenario(),
        "Overcast (sun -60 %)": Scenario(0.4, 1.0, "overcast"),
        "Clear (sun +15 %)": Scenario(1.15, 1.0, "clear"),
        "Heavy hot-water use (+50 %)": Scenario(1.0, 1.5, "heavy use"),
    }[choice]
    if choice == "As forecast":
        runs = {"No backup": b["none_nominal"], "Plain thermostat": b["thermo_nominal"],
                "Planned backup (this page)": b["plan_nominal"]}
    else:
        def go(windows, mode):
            return predictor.simulate(cfg, b["snap"], b["fc"], b["start"], b["steps"], windows, scen, mode)
        runs = {"No backup": go([], "none"), "Plain thermostat": go([], "thermostat"),
                "Planned backup (this page)": go(b["plan"].windows, "plan")}
    rows = []
    for name, df in runs.items():
        m = strategy_metrics(df, b["comfort_wins"], cfg)
        rows.append({
            "Strategy": name,
            "Backup (kWh)": round(m["backup_kwh"], 2),
            "Cost": round(m["cost"], 2),
            "Demand windows below comfort": f"{m['windows_below']} of {m['windows_total']}",
            "Lowest S3 in a window (°C)": round(m["min_s3"], 1),
            "Solar share of heat input": f"{m['solar_share']:.0%}",
        })
    return pd.DataFrame(rows)


def show():

    st.title("🤖 Predictive Control")
    st.caption(
        "Forecasts the tank, plans the backup heater and runs it automatically, with manual override. "
        "The plan comes from the digital-twin physics baseline; switch on the AI safety layer in Settings "
        "to let the trained forecaster add heating when a forecast is at risk."
    )

    cfg = _settings_panel()
    st.session_state.pc_cfg = cfg
    predictors = available_predictors()
    predictor = predictors.get(cfg.predictor, TwinPredictor())

    now = _now(cfg)
    snap = acquire_snapshot(cfg)

    # ---- plan (recomputed when inputs change, every 30 min, or on request) ----
    signature = _signature(cfg, snap, now)
    stale = (st.session_state.get("pc_bundle") is None
             or st.session_state.get("pc_sig") != signature
             or now - st.session_state["pc_bundle"]["made_at"] > REPLAN_AFTER)
    top = st.columns([6, 1])
    with top[1]:
        if st.button("↻ Replan", use_container_width=True, key="pc_replan"):
            stale = True
    if stale:
        with st.spinner("Forecasting and planning..."):
            start = _floor(now, 5)
            horizon_h = (datetime.combine(start.date() + timedelta(days=2), dtime(0)) - start).total_seconds() / 3600
            fc = get_forecast(cfg, start, horizon_h)
            st.session_state.pc_bundle = build_bundle(predictor, cfg, snap, fc, now)
            st.session_state.pc_sig = signature
            # --- ADD THIS LINE TO BRIDGE THE PAGES ---
    st.session_state.active_heater_plan = st.session_state.pc_bundle["plan"].windows
    b = st.session_state.pc_bundle
    with top[0]:
        st.caption(f"Tank state: **{snap.origin}** · Weather: **{b['fc'].source}** · "
                   f"Predictor: **{b['predictor']}** · Planned {b['made_at'].strftime('%H:%M')}")
    for text in (snap.note, b["fc"].note):
        if text:
            st.info(text)

    # ---- decision + manual override (auto-refreshing) ----
    st.fragment(run_every=CONTROL_TICK_S)(_controller_panel)()

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    mp, mt = b["m_plan"], b["m_thermo"]
    worst = max([r["p_plan"] for r in b["risk"]], default=0.0)
    worst_none = max([r["p_none"] for r in b["risk"]], default=0.0)
    k1.metric("Backup energy (to end of tomorrow)", f"{mp['backup_kwh']:.1f} kWh",
              f"{mp['backup_kwh'] - mt['backup_kwh']:+.1f} kWh vs thermostat", delta_color="inverse")
    k2.metric("Backup cost", f"{mp['cost']:.2f}",
              f"{mp['cost'] - mt['cost']:+.2f} vs thermostat", delta_color="inverse")
    k3.metric("Solar share of heat input", f"{mp['solar_share']:.0%}")
    k4.metric("Chance of missing comfort", f"{worst:.0%}",
              f"{worst - worst_none:+.0%} vs no backup", delta_color="inverse")

    # ---- forecast + plan ----
    chart_col, plan_col = st.columns([3, 2])
    with chart_col:
        st.subheader("Tank forecast")
        _show_html(forecast_svg(b, cfg, now), height=430)
        st.caption("Orange band: 10-90 % range of S3 over weather and usage scenarios. "
                   "Hover the line for hourly values.")
    with plan_col:
        st.subheader("Comfort risk")
        if not b["risk"]:
            st.write("No demand windows in the forecast horizon.")
        for r in b["risk"]:
            w = r["window"]
            label = f"{w.label} {_day_hm(w.start, b['start'])}-{_hm(w.end)}"
            st.progress(min(max(r["p_none"], 0.0), 1.0), text=f"{label}: {r['p_none']:.0%} risk without backup")
            st.progress(min(max(r["p_plan"], 0.0), 1.0), text=f"{label}: {r['p_plan']:.0%} risk with the plan")
        st.subheader("Backup plan")
        if b["plan_rows"]:
            st.dataframe(pd.DataFrame([{
                "For": row["window"].label,
                "From": _day_hm(row["window"].start, b["start"]),
                "To": _hm(row["window"].end),
                "Minutes": int((row["window"].end - row["window"].start).total_seconds() // 60),
                "kWh": round(row["kwh"], 2),
                "Cost": round(row["cost"], 2),
            } for row in b["plan_rows"]]), hide_index=True, use_container_width=True)
        else:
            st.success("No backup heating needed in the forecast.")

    _ai_status_panel()

    # ---- detail tabs ----
    t_ai, t_energy, t_what, t_why, t_perf, t_data, t_log = st.tabs([
        "🤖 AI closed-loop test", "⚡ Energy balance", "🔀 What-if", "🧠 Why this plan",
        "📈 Model performance", "🔌 Data and constraints", "🧾 Decision log",
    ])

    with t_ai:
        render_ai_tab()

    with t_energy:
        left, right = st.columns([3, 2])
        with left:
            _show_html(energy_svg(b["days"]), height=320)
        with right:
            st.markdown("**Weather inputs**")
            fc = b["fc"]
            df_w = pd.DataFrame({"time": fc.times, "Irradiance (W/m²)": fc.irradiance.round(0),
                                 "Ambient (°C)": fc.ambient.round(1)})
            st.dataframe(df_w[df_w["time"].dt.hour % 3 == 0].head(16), hide_index=True,
                         use_container_width=True, height=250)

    with t_what:
        choice = st.radio("Weather / usage scenario", ["As forecast", "Overcast (sun -60 %)",
                          "Clear (sun +15 %)", "Heavy hot-water use (+50 %)"], horizontal=True, key="pc_scenario")
        table = _whatif_table(b, cfg, predictor, choice)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("Strategy")["Backup (kWh)"])
        st.caption("The planned heater windows stay as scheduled; only the weather or usage changes. "
                   "That shows how robust the plan is.")

    with t_why:
        for reason in b["reasons"]:
            st.markdown("- " + reason)
        st.info("Feature importance of the AI model (for example SHAP values) will appear here once "
                "the trained model is connected.")

    with t_perf:
        st.info("No forecast history yet. Once predictions and the sensor readings that follow them are "
                "stored, this tab will show:")
        st.markdown(
            "- forecast vs actual S3 / S4 for the last 24 h\n"
            "- error by forecast horizon (1 h, 3 h, 6 h, 12 h, 24 h)\n"
            "- how often reality fell inside the 10-90 % range\n"
            "- backup runs that turned out to be unnecessary\n"
            "- drift warning and last retrain date"
        )

    with t_data:
        rows = [
            ("S1-S4 tank and collector sensors", snap.origin, "ok" if snap.origin != SRC_TWIN or snap.note == "" else "default values"),
            ("Weather forecast", b["fc"].source, "ok" if not b["fc"].note.startswith(("Enter", "Open-Meteo failed")) else "fallback"),
            ("Hot-water draw", "assumed daily profile", f"{cfg.daily_use_l:.0f} L/day, not measured"),
            ("Heat meters", "not used in live telemetry", "estimated by the model"),
            ("Heater power", "setting", f"{cfg.heater_power_w:.0f} W"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Input", "Source", "Status"]), hide_index=True,
                     use_container_width=True)
        st.markdown("**Constraints always enforced by the controller**")
        st.markdown(
            f"- heater OFF above S3 = {cfg.heater_inhibit_c:.0f} °C (also in manual override)\n"
            f"- planned runs stop when S2 reaches {cfg.heater_target_c:.0f} °C\n"
            f"- minimum {MIN_CYCLE_MIN} min on / off between automatic switching\n"
            f"- manual override expires on its own and returns to AUTO\n"
            "- planned: periodic high-temperature cycle (legionella), if you need one"
        )

    with t_log:
        log = st.session_state.get("pc_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)
            if st.button("Clear log", key="pc_clear_log"):
                st.session_state.pc_log = []
                st.rerun()
        else:
            st.write("No decisions logged yet.")

    # =====================================================
    # NAVIGATION
    # =====================================================
    if st.button("⬅ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()