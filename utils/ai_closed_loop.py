"""
STMaCS - AI CLOSED-LOOP TEST ON THE DIGITAL TWIN

What it does
------------
Plays the digital twin forward over a few real days (real-time scale, so the
model's 1-2 hour horizons mean what they were trained to mean) under the same
weather and the same hot-water use, three times:

  1. No backup heater
  2. Plain thermostat (the twin's own thermostat on the middle sensor, S2)
  3. AI controller: every 5 minutes the sensors feed the feature builder, the
     safety forecaster predicts the tank top (S3) 1 and 2 hours ahead, and the
     rule-based controller decides whether the heater should run.

It reports, for each strategy, how much backup energy was used and how much
hot water was delivered colder than the comfort limit.

For the first hour the AI controller has too little history for the model, so
it runs as a plain thermostat (the same fallback it would use after a power-up).

Files needed next to this one (all in utils/ in the project):
  digital_twin_model.py, stmacs_features.py, stmacs_forecaster.py,
  stmacs_controller.py, plus the trained models folder (models/twin_v1).

In the app:   from utils.ai_closed_loop import render_tab
              ... with some_tab:  render_tab()
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:                                           # package layout used by the app
    from utils.digital_twin_model import HOTSETDigitalTwin
    from utils.stmacs_features import FeatureBuilder, InsufficientHistory
    from utils.stmacs_forecaster import SafeForecaster
    from utils.stmacs_controller import Controller, ControllerConfig, HeaterModel
except ImportError:                            # flat layout (scripts / tests)
    from digital_twin_model import HOTSETDigitalTwin
    from stmacs_features import FeatureBuilder, InsufficientHistory
    from stmacs_forecaster import SafeForecaster
    from stmacs_controller import Controller, ControllerConfig, HeaterModel

COMFORT_C = 45.0
SAMPLE_MIN = 5
FORCE_ON_SETPOINT_C = 95.0          # twin thermostat setpoint that never cuts the element
WEATHER_TYPES = ("clear", "partly", "overcast")

_FALLBACK_HOURLY = np.array([
    0.5, 0.3, 0.2, 0.2, 0.4, 1.5, 4.0, 5.0, 3.0, 1.5, 1.0, 1.0,
    1.5, 1.0, 0.8, 0.8, 1.2, 3.0, 5.0, 5.0, 4.0, 2.5, 1.5, 0.8])


# =============================================================================
# SCENARIO
# =============================================================================

@dataclass
class Scenario:
    start: datetime = datetime(2026, 7, 6, 0, 0)       # a winter Monday
    weather: tuple = ("clear", "overcast", "partly")   # one entry per day
    daily_use_l: float = 150.0
    ambient_offset_c: float = 0.0
    heater_w: float = 5000.0
    thermostat_c: float = 50.0                         # baseline thermostat (on S2)
    extra_cushion_2h_c: float = 2.0                    # extra margin on the 2 h forecast
    off_buffer_c: float = 3.0                          # may stop once S3 >= 45 + this ...
    hysteresis_c: float = 1.5                          # ... and forecasts clear by this
    seed: int = 7


def make_inputs(scen: Scenario):
    """Per-minute irradiance, ambient, mains temperature and hot-water draw (L/min)."""
    rng = np.random.default_rng(scen.seed)
    n_days = len(scen.weather)
    n = n_days * 1440
    ts = pd.date_range(scen.start, periods=n, freq="min")
    doy = ts.dayofyear.to_numpy().astype(float)
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    minutes = np.arange(n)
    day_idx = minutes // 1440

    ang = 2 * np.pi * (doy - 355) / 365.25
    daylength = 12.1 + 1.6 * np.cos(ang)
    g_max = 920 + 80 * np.cos(ang)
    x = (hour - (12 - daylength / 2)) / daylength
    clear = np.where((x > 0) & (x < 1), g_max * np.sin(np.pi * np.clip(x, 0, 1)) ** 1.15, 0.0)

    params = {"clear": (0.97, 0.03, 0.85), "partly": (0.60, 0.22, 0.12),
              "overcast": (0.20, 0.08, 0.05)}
    mean_f = np.array([params[w][0] for w in scen.weather])[day_idx]
    sd_f = np.array([params[w][1] for w in scen.weather])[day_idx]
    low_f = np.array([params[w][2] for w in scen.weather])[day_idx]
    n5 = n // 5 + 3
    ar, eps = np.zeros(n5), rng.standard_normal(n5)
    for i in range(1, n5):
        ar[i] = 0.97 * ar[i - 1] + math.sqrt(1 - 0.97 ** 2) * eps[i]
    noise = np.interp(minutes / 5.0, np.arange(n5), ar)
    irradiance = clear * np.clip(mean_f + sd_f * noise, low_f, 1.0)

    season = np.cos(2 * np.pi * (doy - 15) / 365.25)
    ambient = (19.5 + 6.5 * season + scen.ambient_offset_c
               + 7.0 * np.cos(2 * np.pi * (hour - 15) / 24))
    cold = 22.0 + 4.0 * season

    draw = np.zeros(n + 240)
    share = _FALLBACK_HOURLY / _FALLBACK_HOURLY.sum()
    events_per_day = scen.daily_use_l / 25.0
    for d in range(n_days):
        lam = events_per_day * rng.lognormal(-0.045, 0.3)
        for h in range(24):
            for _ in range(rng.poisson(lam * share[h])):
                volume = float(np.clip(rng.lognormal(math.log(25.0) - 0.125, 0.5), 8.0, 80.0))
                flow = rng.uniform(4.0, 9.0)
                minute = int(d * 1440 + h * 60 + rng.uniform(0, 60))
                remaining = volume
                while remaining > 1e-9 and minute < len(draw):
                    use = min(flow, remaining)
                    draw[minute] += use
                    remaining -= use
                    minute += 1
    return irradiance, ambient, cold, draw[:n]


# =============================================================================
# SIMULATION
# =============================================================================

def _driven_twin_class():
    class DrivenTwin(HOTSETDigitalTwin):
        def __init__(self, **kwargs):
            self._irr_now = 0.0
            super().__init__(**kwargs)

        def calculate_solar_irradiance(self, simulation_time_seconds):
            return self._irr_now
    return DrivenTwin


@dataclass
class Result:
    series: dict = field(default_factory=dict)       # name -> DataFrame (5-min rows)
    metrics: pd.DataFrame = None
    decisions: pd.DataFrame = None                   # AI controller state changes
    scenario: Scenario = None


def _simulate(strategy, scen, inputs, forecaster, controller_config):
    irr, amb, cold, draw = inputs
    Twin = _driven_twin_class()
    twin = Twin(initial_tank_temperature=50.0, ambient_temperature=float(amb[0]),
                cold_water_temperature=float(cold[0]), heater_power=0.0)
    twin.node_temperatures = np.linspace(55.0, 40.0, twin.tank_nodes)   # stratified start
    twin.collector_temperature = 30.0
    twin.update_derived_state()
    initial_useful_kwh = twin.useful_energy_kwh()

    rng = np.random.default_rng(scen.seed + 1000)            # sensor noise for the AI
    fb = FeatureBuilder()
    ctrl = Controller(HeaterModel(power_kw=scen.heater_w / 1000.0, volume_l=300.0),
                      controller_config)
    cmd_on = False
    rows, decisions = [], []
    energy_wh = cold_litres = below_minutes = 0.0
    solar_wh = 0.0
    switches = 0
    last_on = False
    n = len(irr)

    for k in range(n):
        twin._irr_now = float(irr[k])
        twin.set_ambient_temperature(amb[k])
        twin.cold_water_temperature = float(cold[k])
        twin.set_hot_water_demand(draw[k] / 60.0)

        if strategy == "No backup":
            twin.set_heater(0.0, 55.0)
        elif strategy == "Thermostat":
            twin.set_heater(scen.heater_w, scen.thermostat_c)
        else:                                               # AI controller
            twin.set_heater(scen.heater_w if cmd_on else 0.0,
                            FORCE_ON_SETPOINT_C if cmd_on else 55.0)
        twin.step(60)

        running = twin.heater_on and twin.heater_power > 0
        if running:
            energy_wh += twin.heater_power / 60.0
        solar_wh += max(twin.solar_power, 0.0) / 60.0
        if twin.top_temperature < COMFORT_C:
            below_minutes += 1
            cold_litres += draw[k]
        if running != last_on:
            switches += 1
            last_on = running

        if (k + 1) % SAMPLE_MIN == 0:
            now = scen.start + timedelta(minutes=k + 1)
            top, mid, bot = twin.top_temperature, twin.middle_temperature, twin.bottom_temperature
            row = {"time": now, "S1": twin.collector_temperature, "S2": mid, "S3": top,
                   "S4": bot, "heater_on": int(running), "draw_lpm": draw[k],
                   "irradiance": irr[k], "backup_kwh": energy_wh / 1000.0,
                   "pump_on": int(twin.pump_on)}

            if strategy == "AI controller":
                noisy = [round(v + rng.normal(0, 0.1), 1) for v in (top, mid, bot, twin.collector_temperature)]
                fb.update(now, noisy[0], noisy[1], noisy[2], noisy[3],
                          twin.flow_rate * 60.0, int(twin.pump_on))
                try:
                    forecast = forecaster.predict(fb.features(), hour=now.hour,
                                                  horizons=(1, 2), expected=False)
                except InsufficientHistory:
                    forecast = None
                decision = ctrl.step(now, noisy[0], forecast, buffer_centre=noisy[1])
                if decision.heater_on != cmd_on:
                    decisions.append({"time": now, "heater": "ON" if decision.heater_on else "OFF",
                                      "mode": decision.mode, "S3": round(top, 1),
                                      "S2": round(mid, 1), "reason": decision.reason})
                cmd_on = decision.heater_on
                if forecast:
                    row["ai_pessimistic_1h"] = float(forecast[1]["pessimistic_top"][0])
                    row["ai_pessimistic_2h"] = float(forecast[2]["pessimistic_top"][0])
            rows.append(row)

    df = pd.DataFrame(rows)
    total_h = n / 60.0
    metrics = {
        "Strategy": strategy,
        "Backup energy (kWh)": round(energy_wh / 1000.0, 2),
        "Tank energy gained (kWh)": round(twin.useful_energy_kwh() - initial_useful_kwh, 2),
        "Hot water drawn below comfort (L)": round(cold_litres, 1),
        "Hours with S3 below comfort": round(below_minutes / 60.0, 1),
        "Share of time below comfort": f"{below_minutes / 60.0 / total_h:.0%}",
        "Heater switch-overs": switches,
        "Solar share of heat input": f"{solar_wh / (solar_wh + energy_wh):.0%}"
                                     if (solar_wh + energy_wh) > 0 else "n/a",
    }
    return df, metrics, pd.DataFrame(decisions)


def run_closed_loop(scen: Scenario, model_dir) -> Result:
    forecaster = SafeForecaster(model_dir)
    for h in (1, 2):
        if h not in forecaster.horizons:
            raise ValueError(f"The model folder has no {h}-hour model.")
    inputs = make_inputs(scen)
    cfg = ControllerConfig(extra_cushion={2: scen.extra_cushion_2h_c},
                           off_buffer=scen.off_buffer_c, hysteresis=scen.hysteresis_c)

    result = Result(scenario=scen)
    metric_rows = []
    for name in ("No backup", "Thermostat", "AI controller"):
        df, metrics, decisions = _simulate(name, scen, inputs, forecaster, cfg)
        result.series[name] = df
        metric_rows.append(metrics)
        if name == "AI controller":
            result.decisions = decisions
    result.metrics = pd.DataFrame(metric_rows)
    return result


# =============================================================================
# STREAMLIT TAB
# =============================================================================

def _default_model_dir():
    return Path(__file__).resolve().parents[1] / "models" / "twin_v1"


def render_tab():
    import streamlit as st

    st.subheader("AI closed-loop test on the digital twin")
    st.caption(
        "The twin plays several real days (real-time scale) under identical weather and "
        "hot-water use, three times: no backup, a plain thermostat, and the AI controller. "
        "The AI model was trained on data generated by this twin, so this tests the "
        "whole pipeline inside the twin, not accuracy on a real installation."
    )

    model_dir = st.text_input("Trained model folder", str(_default_model_dir()), key="cl_model_dir")

    c1, c2, c3 = st.columns(3)
    with c1:
        start_date = st.date_input("Start date (sets the season)", datetime(2026, 7, 6).date(),
                                   key="cl_start")
        days = st.slider("Days to simulate", 1, 4, 3, key="cl_days")
    with c2:
        weather = [st.selectbox(f"Day {i + 1} weather", WEATHER_TYPES,
                                index=(0, 2, 1, 0)[i], key=f"cl_wx{i}") for i in range(days)]
        use = st.slider("Hot water per day (L)", 50, 300, 150, 10, key="cl_use")
    with c3:
        offset = st.slider("Ambient temperature offset (C)", -8.0, 8.0, 0.0, 0.5, key="cl_amb")
        heater = st.slider("Backup heater power (W)", 1000, 5000, 5000, 500, key="cl_heater")
        cushion = st.slider("Extra safety margin on the 2 h forecast (C)", 0.0, 4.0, 2.0, 0.5,
                            key="cl_cushion")

    if st.button("▶ Run closed-loop test", key="cl_run", type="primary"):
        scen = Scenario(start=datetime.combine(start_date, datetime.min.time()),
                        weather=tuple(weather), daily_use_l=float(use),
                        ambient_offset_c=float(offset), heater_w=float(heater),
                        extra_cushion_2h_c=float(cushion))
        try:
            with st.spinner("Simulating three strategies..."):
                st.session_state.cl_result = run_closed_loop(scen, model_dir)
        except Exception as exc:
            st.error(f"Could not run the test: {exc}")
            st.session_state.pop("cl_result", None)

    res = st.session_state.get("cl_result")
    if res is None:
        st.info("Choose a scenario and press Run. The model folder is created by "
                "train_final_models_v2.py.")
        return

    st.markdown("**Results**")
    st.dataframe(res.metrics, hide_index=True, use_container_width=True)
    st.caption(f"'Below comfort' means S3 (tank top) under {COMFORT_C:.0f} C. Litres counts "
               "only water actually drawn while S3 was below that limit.")

    index = res.series["AI controller"]["time"]
    s3 = pd.DataFrame({name: df["S3"].to_numpy() for name, df in res.series.items()}, index=index)
    s3["Comfort limit"] = COMFORT_C
    st.markdown("**Tank top temperature (S3)**")
    st.line_chart(s3)

    kwh = pd.DataFrame({name: df["backup_kwh"].to_numpy() for name, df in res.series.items()},
                       index=index)
    st.markdown("**Cumulative backup energy (kWh)**")
    st.line_chart(kwh)

    ai = res.series["AI controller"]
    if "ai_pessimistic_1h" in ai.columns:
        fc = pd.DataFrame({"S3 now": ai["S3"].to_numpy(),
                           "AI 1 h pessimistic forecast": ai["ai_pessimistic_1h"].to_numpy(),
                           "AI 2 h pessimistic forecast": ai["ai_pessimistic_2h"].to_numpy()},
                          index=index)
        st.markdown("**What the AI saw: tank top now vs its pessimistic forecasts**")
        st.line_chart(fc)

    st.markdown("**AI controller decisions (state changes only)**")
    if res.decisions is not None and len(res.decisions):
        st.dataframe(res.decisions, hide_index=True, use_container_width=True)
    else:
        st.write("The AI controller never needed the heater in this scenario.")