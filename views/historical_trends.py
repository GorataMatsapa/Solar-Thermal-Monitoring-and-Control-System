"""
historical_trends.py - STMaCS Historical Analytics

Two data sources (chosen on the page):

  * Real-time  -> Firebase `site/history` (every reading the ESP32 pushed), filtered by period.
  * Simulation -> a saved Digital Twin run from Firebase `sim/runs/<run-id>` (written by the Digital
                  Twin page in SIMULATION mode). HIL data is deliberately NOT mixed in here.

All Firebase parsing lives in utils/site_data.py.

Firebase Realtime DB rules need an index for the time-range query:
    "site": { "history": { ".indexOn": ["ts"] } }

Without Firebase connected (st.session_state.db empty) the page runs on demo data.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils import site_data as sd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TZ = sd.TZ

MAX_GAP_S = 300              # gaps longer than this are "no data" and are NOT integrated into energy/runtime
PUMP_FLOW_THRESHOLD = 0.2    # L/min: flow above this counts as pump running (if pump_status is missing)
FLUID_CP = 3.8               # kJ/(kg*K) - glycol/water mix in the collector loop (adjust to your fluid)
FLUID_DENSITY = 1.03         # kg/L
HEATER_RATED_W = 2000        # only used if the log has heater_on (true/false) but no heater_power_w

SENSOR_VIEW = "📊 All Sensors (S1–S4)"
SENSOR_MAX_POINTS = 6000     # above this many readings the sensor graph is averaged into equal buckets

VIEWS = [
    "Overview",
    SENSOR_VIEW,
    "🔥 Collector & Environment",
    "🌡️ Tank Stratification",
    "💧 Hydraulics",
    "⚡ Backup Heater & Energy",
]

LABELS = {
    "S1": "Collector (S1)", "S2": "Tank middle (S2)", "S3": "Tank top (S3)", "S4": "Tank bottom (S4)",
    "Tank_avg": "Average tank", "Ambient": "Outdoor temp", "Irradiance": "Irradiance (W/m²)",
    "Flow": "Flow rate (L/min)", "Spread": "S3 − S4 spread (K)",
    "Solar_kWh": "Solar gain (est., kWh)", "Heater_kWh": "Backup heater (kWh)",
    "Pump_h": "Pump runtime (h)", "Volume_L": "Circulated volume (L)",
}

RAW_COLS = ["S1", "S2", "S3", "S4", "Ambient", "Irradiance", "Flow", "Pump", "Heater_W", "Heater_on"]
MEAN_COLS = ["S1", "S2", "S3", "S4", "Tank_avg", "Ambient", "Irradiance", "Flow", "Spread", "Collector_dT"]
SUM_COLS = ["Solar_kWh", "Heater_kWh", "Pump_h", "Volume_L"]


# ---------------------------------------------------------
# Data layer
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _mock(start_epoch, end_epoch):
    """Deterministic simulated installation: daily sun cycle, tank dynamics, draws, a data gap."""
    idx = pd.date_range(
        pd.to_datetime(start_epoch, unit="s", utc=True),
        pd.to_datetime(end_epoch, unit="s", utc=True),
        freq="300s",
    )
    n = len(idx)
    rng = np.random.default_rng(42)
    local = idx.tz_convert(TZ)
    hour = np.asarray(local.hour + local.minute / 60)
    day = np.asarray((idx - idx[0]).days)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1)
    cloud = 0.7 + 0.3 * np.sin(day * 1.7)
    irr = np.where(sun > 0, np.clip(950 * sun * cloud + rng.normal(0, 25, n), 0, None), 0.0)
    amb = 20 + 7 * np.sin(2 * math.pi * (hour - 9) / 24) + rng.normal(0, 0.5, n)

    top, mid, bot = 45.0, 38.0, 25.0
    out = {k: np.zeros(n) for k in ("S1", "S2", "S3", "S4", "Flow", "Pump", "Heater_W")}
    for i in range(n):
        coll = amb[i] + 55 * sun[i] * cloud[i]
        pump = coll - bot > 8
        if pump:
            top += 0.05 * (coll - 8 - top)
            mid += 0.04 * (coll - 15 - mid)
            bot += 0.03 * (coll - 25 - bot)
        draw = 6.5 <= hour[i] < 7 or 18.5 <= hour[i] < 19.5
        if draw:
            top -= 0.8; mid -= 0.6; bot += 0.4 * (18 - bot)
        heater = 4.5 <= hour[i] < 6.5 and top < 50
        if heater:
            top += 1.0
        top, mid, bot = min(max(top, mid), 90), max(mid, bot), bot
        top += 0.002 * (amb[i] - top); mid += 0.002 * (amb[i] - mid); bot += 0.002 * (amb[i] - bot)
        out["S1"][i] = coll + rng.normal(0, 1)
        out["S3"][i], out["S2"][i], out["S4"][i] = top, mid, bot
        out["Pump"][i] = float(pump)
        out["Flow"][i] = (1.2 + rng.normal(0, 0.08)) if pump else 0.0
        out["Heater_W"][i] = 2000.0 if heater else 0.0

    df = pd.DataFrame(out, index=idx)
    df["Ambient"], df["Irradiance"], df["Heater_on"] = amb, irr, np.nan
    keep = ~((day == 1) & (hour >= 13) & (hour < 17))  # simulate the caravan being unplugged
    return df[keep]


def add_derived(df):
    """Adds energy/runtime/volume by integrating over real sample spacing (gaps are NOT filled)."""
    df = df.copy()
    dt = df.index.to_series().diff().shift(-1).dt.total_seconds()
    dt = dt.where(dt <= MAX_GAP_S, 0).fillna(0)

    flow = df["Flow"].fillna(0)
    pump_on = (df["Pump"].fillna(0) > 0) | (df["Pump"].isna() & (flow >= PUMP_FLOW_THRESHOLD))

    d_t = (df["S1"] - df["S4"]).clip(lower=0).fillna(0)
    kw = flow / 60 * FLUID_DENSITY * FLUID_CP * d_t           # kW = kg/s * kJ/(kg*K) * K
    df["Dt_s"] = dt
    df["Solar_kWh"] = np.where(pump_on, kw, 0.0) * dt / 3600
    df["Pump_h"] = pump_on.astype(float) * dt / 3600
    df["Volume_L"] = flow * dt / 60

    if df["Heater_W"].notna().any():
        power_w = df["Heater_W"].fillna(0)
    elif df["Heater_on"].notna().any():
        power_w = df["Heater_on"].fillna(0) * HEATER_RATED_W
    else:
        power_w = pd.Series(np.nan, index=df.index)
    df["Heater_kWh"] = power_w / 1000 * dt / 3600

    df["Tank_avg"] = df[["S2", "S3", "S4"]].mean(axis=1)
    df["Spread"] = df["S3"] - df["S4"]
    df["Collector_dT"] = df["S1"] - df["Ambient"]
    return df


def resample(df, span_hours):
    """Picks a resolution from the span (never finer than the logging interval) and aggregates."""
    base = 60 if span_hours <= 6 else 600 if span_hours <= 48 else 3600 if span_hours <= 24 * 14 else 21600
    median_dt = df.index.to_series().diff().dt.total_seconds().median()
    step = int(max(base, median_dt if pd.notna(median_dt) else base))
    rule = f"{step}s"
    mean = df[MEAN_COLS].resample(rule).mean()
    total = df[SUM_COLS].resample(rule).sum(min_count=1)
    out = mean.join(total)
    out.index = out.index.tz_convert(TZ)
    return out, step


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------
def _has(df, col):
    return col in df.columns and df[col].notna().any()


def _plot(df, cols, kind="line", height=300):
    cols = [c for c in cols if _has(df, c)]
    if not cols:
        st.caption("No data available for this chart in the selected period.")
        return
    data = df[cols].rename(columns=LABELS)
    {"line": st.line_chart, "area": st.area_chart, "bar": st.bar_chart}[kind](data, height=height)


def _sensor_frame(df):
    """S1-S4 at the finest detail the period allows, with real gaps left as breaks in the lines.

    Up to SENSOR_MAX_POINTS readings: every recorded value is plotted. Beyond that the readings are
    averaged into equal buckets (empty buckets stay empty, so outages still show as gaps).
    Returns (frame indexed by local time, note about the resolution).
    """
    cols = [c for c in ("S1", "S2", "S3", "S4") if _has(df, c)]
    d = df[cols].copy()
    if d.empty:
        return d, ""
    if len(d) > SENSOR_MAX_POINTS:
        span = (d.index[-1] - d.index[0]).total_seconds()
        step = max(int(math.ceil(span / SENSOR_MAX_POINTS)), 1)
        d = d.resample(f"{step}s").mean()
        note = f"{len(df):,} readings averaged into {step // 60 if step >= 60 else step} {'min' if step >= 60 else 's'} buckets"
    else:
        gap = d.index.to_series().diff().dt.total_seconds()
        breaks = d.index[gap > MAX_GAP_S] - pd.Timedelta(seconds=1)   # NaN row just before a late sample = line break
        if len(breaks):
            d = pd.concat([d, pd.DataFrame(np.nan, index=breaks, columns=cols)]).sort_index()
        note = f"all {len(df):,} recorded readings plotted"
    d.index = d.index.tz_convert(TZ)
    return d, note


def _select_range():
    now = pd.Timestamp.now(tz=TZ)
    choice = st.radio("Period", ["Last 24 h", "Last 7 days", "Last 30 days", "Custom"], horizontal=True, key="trend_period")
    if choice == "Custom":
        picked = st.date_input("Date range", value=(now.date() - pd.Timedelta(days=7), now.date()), key="trend_dates")
        if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
            st.info("Pick an end date to load the range.")
            st.stop()
        start = pd.Timestamp(picked[0], tz=TZ)
        end = pd.Timestamp(picked[1], tz=TZ) + pd.Timedelta(days=1)
    else:
        days = {"Last 24 h": 1, "Last 7 days": 7, "Last 30 days": 30}[choice]
        end = now
        start = now - pd.Timedelta(days=days)
    # Round so the cache key is stable between reruns
    return int(start.timestamp() // 300 * 300), int(end.timestamp() // 300 * 300) + 300


# ---------------------------------------------------------
# Page entry point (called from app.py)
# ---------------------------------------------------------
def show():
    col_head1, col_head2 = st.columns([6, 1])
    with col_head1:
        st.title("📈 STMaCS Historical Analytics")
        st.caption("Performance logs and system telemetry trends")
    with col_head2:
        if st.button("⬅ Home"):
            st.session_state.page = "Home"
            st.rerun()

    st.markdown("---")

    data_mode = st.radio("Historical Data Source", ["Real-time (Firebase site/history)", "Simulation"],
                         horizontal=True, key="trend_source")
    is_sim = data_mode == "Simulation"
    st.markdown("---")

    db = sd.get_db()

    # ---- Load ----
    if not is_sim:
        start_epoch, end_epoch = _select_range()
        if db:
            try:
                raw = sd.load_history(db, start_epoch, end_epoch)
            except Exception as e:
                st.error(f"Couldn't load history from Firebase: {e}")
                st.stop()
            if raw.empty:
                st.warning("No logged data found under `site/history` for this period. "
                           "Pick a longer period, or check that the installation was online and the `ts` index rule is set.")
                st.stop()
            raw = sd.sanitise(raw)
            if raw["Ambient"].isna().all():                 # site/history carries no weather: join the weather log
                raw = sd.attach_weather(raw, sd.load_weather(db, start_epoch, end_epoch))
        else:
            st.warning("Firebase is not connected" + (f" ({sd.last_error()})" if sd.last_error() else "") + ". Showing DEMO data, not real measurements.")
            raw = _mock(start_epoch, end_epoch)
    else:
        runs = sd.list_sim_runs(db) if db else []
        if runs:
            labels = {
                r["id"]: f"{r.get('label') or r['id']} · {sd.fmt_local(r.get('started'), '%d %b %H:%M')} · {r.get('samples', '?')} min"
                for r in runs
            }
            run_id = st.selectbox("Simulation run", list(labels), format_func=labels.get, key="trend_sim_run")
            raw = sd.load_sim_run(db, run_id)
            if raw.empty:
                st.warning("That run has no samples.")
                st.stop()
            meta = next(r for r in runs if r["id"] == run_id)
            if meta.get("note"):
                st.caption(meta["note"])
        else:
            st.warning("No saved simulation runs found (`sim/runs`). Run the Digital Twin in SIMULATION mode "
                       "and press Pause or let it finish - the run is saved automatically. Showing DEMO data meanwhile.")
            now = int(pd.Timestamp.now(tz=TZ).timestamp())
            raw = _mock(now - 86400 * 3, now)
        start_epoch, end_epoch = int(raw.index[0].timestamp()), int(raw.index[-1].timestamp()) + 60

    st.caption("Source: " + ("saved Digital Twin simulation - NOT physical measurements." if is_sim else
                             "real installation (site/history)."))
    span_hours = max((end_epoch - start_epoch) / 3600, 1 / 60)

    df = add_derived(raw)
    rs, step = resample(df, span_hours)

    # ---- Headline numbers (computed on raw data, not on resampled averages) ----
    solar_kwh = df["Solar_kWh"].sum()
    heater_kwh = df["Heater_kWh"].sum() if _has(df, "Heater_kWh") else None
    share = solar_kwh / (solar_kwh + heater_kwh) * 100 if heater_kwh is not None and (solar_kwh + heater_kwh) > 0 else None
    coverage = min(df["Dt_s"].sum() / max(end_epoch - start_epoch, 1), 1.0) * 100

    st.caption(
        f"Resolution: {step // 60 if step >= 60 else step} {'min' if step >= 60 else 's'} per point · "
        f"data coverage {coverage:.0f}% of the period (gaps = node off or offline, never interpolated)"
    )

    view = st.radio("Analytical view", VIEWS, horizontal=True, key="trend_view")
    st.markdown("---")

    # ---------------- Overview ----------------
    if view == "Overview":
        st.subheader("📋 Period summary")
        k1, k2, k3 = st.columns(3)
        k1.metric("Solar gain (est.)", f"{solar_kwh:.1f} kWh", help="Flow × cp × (S1 − S4), only while the pump runs. An estimate - no heat meter.")
        k2.metric("Backup heater", "—" if heater_kwh is None else f"{heater_kwh:.1f} kWh")
        k3.metric("Solar share of heat input (est.)", "—" if share is None else f"{share:.0f} %")
        k4, k5, k6 = st.columns(3)
        k4.metric("Pump runtime", f"{df['Pump_h'].sum():.1f} h")
        k5.metric("Peak collector (S1)", f"{df['S1'].max():.1f} °C" if _has(df, "S1") else "—")
        k6.metric("Peak tank top (S3)", f"{df['S3'].max():.1f} °C" if _has(df, "S3") else "—")

        st.markdown("##### Solar gain vs backup heater energy")
        _plot(rs, ["Solar_kWh", "Heater_kWh"], kind="bar")

    # ---------------- All sensors ----------------
    elif view == SENSOR_VIEW:
        st.subheader("📊 S1 – S4 temperatures over the period")
        sensors, note = _sensor_frame(df)
        available = [c for c in sensors.columns if sensors[c].notna().any()]
        if not available:
            st.info("No sensor temperatures were recorded in this period.")
        else:
            chosen = st.multiselect("Sensors", available, default=available,
                                    format_func=lambda c: LABELS[c], key="trend_sensors")
            if chosen:
                st.line_chart(sensors[chosen].rename(columns=LABELS), height=420)
                st.caption(f"{note}. Breaks in a line mean no data (node off or offline) - nothing is interpolated.")
                stats = pd.DataFrame({
                    "Min (°C)": df[chosen].min(),
                    "Average (°C)": df[chosen].mean(),
                    "Max (°C)": df[chosen].max(),
                }).round(1)
                stats.index = [LABELS[c] for c in stats.index]
                st.dataframe(stats, use_container_width=True)
            else:
                st.caption("Select at least one sensor.")

    # ---------------- Collector ----------------
    elif view == "🔥 Collector & Environment":
        st.subheader("🔥 Collector performance vs environment")
        _plot(rs, ["S1", "Ambient"])
        st.markdown("##### Solar irradiance (estimated, weather API)")
        _plot(rs, ["Irradiance"], kind="area", height=220)
        if _has(rs, "Irradiance") and _has(rs, "Collector_dT"):
            st.markdown("##### Collector temperature rise vs irradiance")
            sc = rs.loc[rs["Irradiance"] > 50, ["Irradiance", "Collector_dT"]].dropna()
            sc.columns = ["Irradiance (W/m²)", "Collector above outdoor (K)"]
            if len(sc) > 5:
                st.scatter_chart(sc, x="Irradiance (W/m²)", y="Collector above outdoor (K)", height=300)
                st.caption("Each dot is one time bucket. A cloud that drifts low means a fouled collector, air in the loop or the pump not running.")

    # ---------------- Tank ----------------
    elif view == "🌡️ Tank Stratification":
        st.subheader("🌡️ Tank temperature layers")
        _plot(rs, ["S3", "S2", "S4", "Tank_avg"])
        st.markdown("##### Stratification spread (top − bottom)")
        _plot(rs, ["Spread"], height=220)
        st.caption("A healthy tank holds a large spread; it collapses when the tank is fully mixed or fully charged.")

    # ---------------- Hydraulics ----------------
    elif view == "💧 Hydraulics":
        st.subheader("💧 Collector loop circulation")
        _plot(rs, ["Flow"], height=240)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Pump runtime per period")
            _plot(rs, ["Pump_h"], kind="bar", height=240)
        with c2:
            st.markdown("##### Circulated volume")
            _plot(rs, ["Volume_L"], kind="bar", height=240)
        st.caption("This is circulation in the collector loop. Domestic hot-water draw isn't measured by the current sensors.")

    # ---------------- Heater ----------------
    elif view == "⚡ Backup Heater & Energy":
        st.subheader("⚡ Backup heater and solar offset")
        if heater_kwh is None:
            st.info("No heater data in the log (expected `heater_power_w` or `heater_on`). Showing solar gain only.")
        _plot(rs, ["Heater_kWh"], kind="bar", height=260)
        st.markdown("##### Cumulative energy over the period")
        cum = rs[["Solar_kWh", "Heater_kWh"]].fillna(0).cumsum()
        _plot(cum, ["Solar_kWh", "Heater_kWh"])

    # ---- Export ----
    with st.expander("📄 Data table & export"):
        table = rs.rename(columns=LABELS).round(2)
        st.dataframe(table, use_container_width=True)
        st.download_button(
            "⬇ Download CSV", table.to_csv().encode("utf-8"),
            file_name="stmacs_history.csv", mime="text/csv",
        )