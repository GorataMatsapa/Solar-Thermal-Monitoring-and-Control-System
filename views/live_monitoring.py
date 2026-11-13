"""
live_monitoring.py - STMaCS Live Monitoring page

Reads `site/latest` (the newest ESP32 reading) through utils/site_data.py:

    site/latest: {ts, s1_collector, s2_tank_mid, s3_tank_top, s4_tank_bottom, flow_lpm, pulses, volume_l}

The page ALWAYS shows the last recorded reading. If the installation has stopped reporting it is
flagged STALE / OFFLINE and the exact date and time of the last reading is shown.
The trend chart is seeded from `site/history` (measured back from the newest record, so it still
works while the installation is offline) and then extended with live readings.

Needs Streamlit >= 1.37 for st.fragment (auto-refresh of just the live section).
"""

import math
import time
from collections import deque
from datetime import datetime

import pandas as pd
import streamlit as st

from hil import dashboard as hil_panels
from utils import site_data as sd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
REFRESH_SECONDS = 5          # how often the live section re-reads Firebase
STALE_AFTER_S = 60           # older than this -> "stale" warning
OFFLINE_AFTER_S = 180        # older than this -> "offline"
HISTORY_MINUTES = 60         # trend chart window
PUMP_ON_DELTA_T = 8.0        # K: collector hotter than tank bottom by this much -> pump should run
NO_FLOW_THRESHOLD = sd.PUMP_FLOW_THRESHOLD
NO_DATA_COLOR = "#9E9E9E"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def get_temp_color(temp):
    """Returns (color, label) based on temperature thresholds for solar thermal systems."""
    if temp is None:
        return NO_DATA_COLOR, "No data"
    if temp < 35.0:
        return "#1E88E5", "Cool / Return"
    elif temp < 75.0:
        return "#43A047", "Optimal"
    elif temp < 90.0:
        return "#FB8C00", "High Temp"
    return "#E53935", "Overheat Alert!"


def fmt(v, unit="", decimals=1):
    return "—" if v is None else f"{v:.{decimals}f}{(' ' + unit) if unit else ''}"


def fmt_age(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h {(seconds % 3600) // 60} min ago"
    return f"{seconds // 86400} d {(seconds % 86400) // 3600} h ago"


def demo_data():
    """Clearly-labelled fake data (same field names as site/latest), only used when Firebase isn't connected."""
    t = time.time()
    return {
        "ts": t,
        "s1_collector": 72 + 12 * math.sin(t / 60),
        "s2_tank_mid": 50 + 2 * math.sin(t / 90),
        "s3_tank_top": 66 + 2 * math.sin(t / 80),
        "s4_tank_bottom": 32 + 1.5 * math.sin(t / 70),
        "flow_lpm": 3.8,
        "volume_l": 1234.5,
        "pulses": 0,
    }


def fetch_snapshot():
    """Returns (raw_dict | None, error_message | None, is_demo)."""
    try:
        db = sd.get_db()
        if not db:
            return demo_data(), None, True
        raw, err = sd.read_latest(db)
        return raw, err, False
    except Exception as e:  # network / HTTP errors must never break the page
        return None, f"{type(e).__name__}: {e}", False


def update_history(d):
    """Rolling in-session buffer for the trend chart (de-duplicated by device timestamp)."""
    hist = st.session_state.setdefault("live_history", deque(maxlen=2000))
    ts = d["epoch"] or time.time()
    if hist and hist[-1]["t"] == ts:
        return hist
    hist.append({"t": ts, "S1": d["s1"], "S3": d["s3"], "S4": d["s4"]})
    # window is measured back from the newest point, so an offline installation still shows its last hour
    cutoff = hist[-1]["t"] - HISTORY_MINUTES * 60
    while hist and hist[0]["t"] < cutoff:
        hist.popleft()
    return hist


def seed_history():
    """Once per session: fill the trend from site/history so the chart isn't empty on page load."""
    if "live_history" in st.session_state:
        return
    hist = deque(maxlen=2000)
    try:
        db = sd.get_db()
        if db:
            for r in sd.read_recent(db, HISTORY_MINUTES):
                hist.append({"t": r["epoch"], "S1": r["s1"], "S3": r["s3"], "S4": r["s4"]})
    except Exception:
        # Connection problem: leave the buffer unset so the next rerun retries seeding
        return
    st.session_state["live_history"] = hist


# ---------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------
def tank_html(s2, s3, s4):
    layers = [("S3 · Top (hot water)", s3), ("S2 · Middle", s2), ("S4 · Bottom (cold)", s4)]
    rows = ""
    for label, val in layers:
        color, _ = get_temp_color(val)
        rows += (
            f"<div style='background:{color}; color:white; padding:16px 10px; text-align:center;'>"
            f"<div style='font-size:0.8em; opacity:0.9;'>{label}</div>"
            f"<div style='font-size:1.5em; font-weight:700;'>{fmt(val, '°C')}</div></div>"
        )
    return (
        "<div style='border:3px solid #78909C; border-radius:28px; overflow:hidden; "
        f"max-width:240px; margin:auto;'>{rows}</div>"
    )


def sensor_card(name, val, desc):
    color, label = get_temp_color(val)
    st.markdown(
        f"""
        <div style="padding:14px; border-radius:10px; border:2px solid {color};
                    background-color:{color}18; margin-bottom:4px;">
            <div style="font-weight:600; opacity:0.85;">{name}</div>
            <div style="font-size:1.8em; font-weight:700; color:{color};">{fmt(val, '°C')}</div>
            <div style="font-size:0.85em; font-weight:600; color:{color};">● {label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(desc)


def collect_alerts(d):
    alerts = []
    s1, s3, s4, flow, pump = d["s1"], d["s3"], d["s4"], d["flow"], d["pump"]
    if d.get("bad_sensors"):
        alerts.append(("error", f"Invalid reading from {', '.join(d['bad_sensors'])} (e.g. -127 °C = sensor "
                                "disconnected). Check wiring - the value is hidden below."))
    if (s1 is not None and s1 >= 90) or (s3 is not None and s3 >= 90):
        alerts.append(("error", "Overheat: a sensor is at or above 90 °C."))
    if s1 is not None and s4 is not None and pump is False and (s1 - s4) >= PUMP_ON_DELTA_T:
        alerts.append((
            "info",
            f"Collector is {s1 - s4:.1f} °C hotter than the tank bottom but no flow is detected "
            "(expected if the controller is waiting or in manual mode; otherwise check pump, air lock or valves).",
        ))
    if not d.get("pump_inferred"):
        # only meaningful when the node reports a real pump flag (otherwise pump is derived from flow)
        if pump is True and flow is not None and flow < NO_FLOW_THRESHOLD:
            alerts.append(("warning", "Pump is ON but flow is near zero - check for an air lock, closed valve or dry run."))
        if pump is False and flow is not None and flow >= NO_FLOW_THRESHOLD:
            alerts.append(("warning", "Flow is detected while the pump reports OFF - check the relay / flow sensor."))
    return alerts


# ---------------------------------------------------------
# Live section (re-runs on its own every REFRESH_SECONDS)
# ---------------------------------------------------------
def _render_live():
    raw, err, is_demo = fetch_snapshot()

    # Keep the last good reading so a dropped connection never blanks the page
    if raw is not None:
        st.session_state["live_last"] = {"data": sd.parse_record(raw), "received": time.time()}
    last = st.session_state.get("live_last")

    if last is None:
        st.error(f"No telemetry received yet. {err or ''}")
        return

    d = last["data"]
    if raw is not None:
        update_history(d)
    if d["ambient"] is None and d["irradiance"] is None:
        try:
            wx = sd.read_weather_latest(sd.get_db())    # site/weather/latest (weather script)
        except Exception:
            wx = None                                   # weather is optional; ignore network errors
        if wx:
            d = dict(d, ambient=wx.get("ambient"), irradiance=wx.get("irradiance"), weather_epoch=wx.get("epoch"))

    # Freshness: trust the device timestamp, fall back to when we received it
    ref_epoch = d["epoch"] or last["received"]
    age = 0 if is_demo else time.time() - ref_epoch
    if is_demo:
        state = "demo"
    elif age <= STALE_AFTER_S:
        state = "live"
    elif age <= OFFLINE_AFTER_S:
        state = "stale"
    else:
        state = "offline"

    last_recorded = sd.fmt_local(ref_epoch)

    # ---- Status bar ----
    c1, c2, c3, c4 = st.columns(4)
    badge = {
        "live": "🟢 **LIVE**",
        "stale": "🟠 **STALE**",
        "offline": "🔴 **OFFLINE**",
        "demo": "🟣 **DEMO DATA**",
    }[state]
    c1.markdown(f"Node: {badge}")
    if d["pump"] is True:
        c2.markdown("Pump: 🔵 **ON**" + (" <small>(from flow)</small>" if d.get("pump_inferred") else ""),
                    unsafe_allow_html=True)
    elif d["pump"] is False:
        c2.markdown("Pump: ⚪ **OFF**" + (" <small>(from flow)</small>" if d.get("pump_inferred") else ""),
                    unsafe_allow_html=True)
    else:
        c2.markdown("Pump: ❔ **UNKNOWN**")
    c3.markdown(f"Mode: ⚡ **{str(d['mode']).upper() if d['mode'] else '—'}**")
    c4.markdown(
        f"<div style='text-align:right; color:gray;'>Last reading {fmt_age(age)}<br>"
        f"<small>{last_recorded}</small></div>",
        unsafe_allow_html=True,
    )

    if state == "demo":
        st.info("Firebase isn't connected - showing simulated values."
                + (f" Reason: {sd.last_error()}" if sd.last_error() else ""))
    elif state == "stale":
        st.warning(f"No new data for {fmt_age(age)} (last reading {last_recorded}). "
                   "The installation may have lost its connection - values below are the last known ones.")
    elif state == "offline":
        st.error(f"Installation offline. Last reading was recorded on {last_recorded} ({fmt_age(age)}). "
                 "Values below are the last known ones, not live.")
    if err and state != "demo":
        st.caption(f"Last Firebase read failed: {err}")

    for level, msg in collect_alerts(d):
        getattr(st, level)(msg)

    st.markdown("---")

    # ---- Core metrics ----
    s1, s2, s3, s4 = d["s1"], d["s2"], d["s3"], d["s4"]
    tank_vals = [v for v in (s2, s3, s4) if v is not None]
    avg_tank = sum(tank_vals) / len(tank_vals) if tank_vals else None
    ambient, irr = d["ambient"], d["irradiance"]

    st.subheader("⚡ System" + ("" if state in ("live", "demo") else " (last recorded)"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Collector (S1)", fmt(s1, "°C"),
        delta=f"{s1 - ambient:.1f} °C vs outdoor" if s1 is not None and ambient is not None else None,
        delta_color="off",
    )
    m2.metric("Avg tank temp", fmt(avg_tank, "°C"), help="Mean of S2, S3 and S4 (only sensors currently reporting).")
    m3.metric("Hot water out (S3)", fmt(s3, "°C"), help="Top of tank - what the user actually draws.")
    m4.metric("Flow rate", fmt(d["flow"], "L/min"))

    st.subheader("📊 Performance" + (" & environment" if ambient is not None or irr is not None else ""))
    dt = s1 - s4 if s1 is not None and s4 is not None else None
    cols = st.columns(4 if (ambient is not None or irr is not None) else 2)
    cols[0].metric("Collector – tank ΔT", fmt(dt, "K"), help="S1 − S4. Larger = more heat available to harvest.")
    cols[1].metric("Total circulated volume", fmt(d["volume"], "L"), help="Cumulative volume counter from the flow meter (volume_l).")
    if ambient is not None or irr is not None:
        cols[2].metric("Outdoor temp", fmt(ambient, "°C"), help="From weather API, not a local sensor.")
        cols[3].metric("Solar irradiance (est.)", fmt(irr, "W/m²", 0), help="Modelled by the weather service, not measured on-site.")
        if d["weather_epoch"]:
            st.caption(f"Weather data fetched {fmt_age(time.time() - d['weather_epoch'])} (estimates, not on-site measurements).")

    st.markdown("---")

    # ---- Sensor array ----
    st.subheader("🌡️ Sensor array (S1 – S4)")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(tank_html(s2, s3, s4), unsafe_allow_html=True)
        st.caption("Tank cross-section (top → bottom)")
    with right:
        r1, r2 = st.columns(2)
        with r1:
            sensor_card("S1 · Collector line", s1, "Hot line out of the solar collector loop.")
            sensor_card("S2 · Tank middle", s2, "Mid-tank thermocline.")
        with r2:
            sensor_card("S3 · Tank top", s3, "Stored hot water reserve.")
            sensor_card("S4 · Tank bottom", s4, "Cold water / collector return.")

    st.markdown("---")

    # ---- Trend ----
    st.subheader(f"📈 Last {HISTORY_MINUTES} minutes" + ("" if state in ("live", "demo") else " before the last reading"))
    hist = st.session_state.get("live_history")
    if hist and len(hist) >= 2:
        df = pd.DataFrame(list(hist))
        df["time"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(sd.TZ)
        df = df.set_index("time")[["S1", "S3", "S4"]]
        df.columns = ["Collector (S1)", "Tank top (S3)", "Tank bottom (S4)"]
        st.line_chart(df, height=280)
        st.caption("Loaded from the history log and extended with live readings. Longer periods are on the Historical Trends page.")
    else:
        st.caption("Not enough history yet - the trend appears after a couple of readings.")


def _live_body():
    """Safety net: any error during a refresh keeps the page up instead of showing an HTTP/exception screen."""
    try:
        _render_live()
    except Exception as e:
        st.warning(f"Live refresh hit a connection problem ({type(e).__name__}). Retrying automatically "
                   f"in {REFRESH_SECONDS} s - the last known reading stays valid.")


# Auto-refresh only this section (not the whole page) when st.fragment is available
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _fragment is not None:
    _live_body = _fragment(run_every=REFRESH_SECONDS)(_live_body)


# ---------------------------------------------------------
# Page entry point (called from app.py)
# ---------------------------------------------------------
def show():
    col_head1, col_head2 = st.columns([6, 1])
    with col_head1:
        st.title("📡 STMaCS Live Monitoring")
        st.caption("Real-time telemetry from the installation (Firebase `site/latest`)")
    with col_head2:
        if st.button("⬅ Home"):
            st.session_state.page = "Home"
            st.rerun()

    st.markdown("---")

    if _fragment is None:
        st.caption("Upgrade Streamlit to 1.37+ for automatic refresh; use the button below for now.")
        if st.button("🔄 Refresh"):
            st.rerun()

    seed_history()
    _live_body()

    st.markdown("---")
    hil_panels.status_strip()      # HIL test status (test data, kept separate from the telemetry above)