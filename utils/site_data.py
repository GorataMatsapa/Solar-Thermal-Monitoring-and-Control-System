"""
utils/site_data.py - ONE place that knows how the Firebase `site` and `sim` nodes are laid out.

Pages (live monitoring, digital twin, historical trends, fault logs) import from here instead of
each re-implementing parsing, so a schema change is a one-file fix.

Firebase layout this module expects
-----------------------------------
site/latest              the newest reading (overwritten by the ESP32)
site/history/<push-id>   every reading, appended (same fields as latest)
    ts            epoch seconds
    s1_collector, s2_tank_mid, s3_tank_top, s4_tank_bottom      deg C
    flow_lpm      L/min
    pulses, volume_l
site/weather/latest      {ts, ambient_temp, irradiance}   written by weather_to_firebase.py
site/weather/history/<id> same fields, appended every run
sim/index/<run-id>       small metadata for each saved digital-twin simulation run
sim/runs/<run-id>        {"meta": {...}, "samples": [ {ts, s1_collector, ... }, ... ]}

Required RTDB rule for the time-range query:
    "site": { "history": { ".indexOn": ["ts"] }, "weather": { "history": { ".indexOn": ["ts"] } } }
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

TZ = "Africa/Johannesburg"
LATEST_PATH = "site/latest"
HISTORY_PATH = "site/history"
WEATHER_LATEST_PATH = "site/weather/latest"
WEATHER_HISTORY_PATH = "site/weather/history"
SIM_INDEX_PATH = "sim/index"
SIM_RUNS_PATH = "sim/runs"

PUMP_FLOW_THRESHOLD = 0.2        # L/min: flow at/above this = pump running (site data has no pump flag)
SENSOR_KEYS = {"S1": "s1_collector", "S2": "s2_tank_mid", "S3": "s3_tank_top", "S4": "s4_tank_bottom"}

RAW_COLS = ["S1", "S2", "S3", "S4", "Ambient", "Irradiance", "Flow", "Pump", "Heater_W", "Heater_on"]

# Anomaly rule settings (experimental - tune against your own data)
OVERHEAT_C = 90.0
HOT_COLLECTOR_DT = 8.0           # K: S1 - S4 above this and no flow = pump should be running
SUSTAIN_S = 600                  # a condition must last this long before it is reported


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def get_db():
    """
    The firebase_admin `db` module, or None. Uses st.session_state.db if something set it; otherwise
    connects the same way hil/dashboard.py does (importing utils.firebase initialises firebase_admin
    with the service-account file). The reason for a failure is kept in last_error().
    """
    db = st.session_state.get("db")
    if db:
        return db
    try:
        import utils.firebase  # noqa: F401  (initialises firebase_admin with your service account)
        from firebase_admin import db as firebase_db
        st.session_state.db = firebase_db
        st.session_state.pop("firebase_error", None)
        return firebase_db
    except Exception as exc:
        st.session_state.firebase_error = f"{type(exc).__name__}: {exc}"
        return None


def last_error():
    return st.session_state.get("firebase_error")


def num(x):
    """Safe float; None for missing / invalid (never a fake 0.0)."""
    try:
        if x is None:
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def to_epoch(ts):
    """Epoch seconds, epoch milliseconds, or 'YYYY-MM-DD HH:MM:SS' string -> epoch seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return ts / 1000.0 if ts > 1e11 else float(ts)
    try:
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def fmt_local(epoch, fmt="%d %b %Y, %H:%M:%S"):
    """Epoch -> text in the project's local time zone (NOT the server's)."""
    if epoch is None:
        return "—"
    return pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert(TZ).strftime(fmt)


def valid_temp(v):
    """DS18B20 returns -127 when disconnected; anything outside this window is a sensor problem."""
    return v is not None and -55.0 <= v <= 125.0


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# ---------------------------------------------------------
# Latest reading
# ---------------------------------------------------------
def read_latest(db):
    """Returns (raw_dict | None, error_message | None)."""
    if not db:
        return None, "Firebase is not connected"
    try:
        raw = db.reference(LATEST_PATH).get()
        if not raw or not isinstance(raw, dict):
            return None, f"Nothing found at '{LATEST_PATH}'"
        return raw, None
    except Exception as e:
        return None, str(e)


def parse_record(r):
    """
    One site/latest (or history) record -> clean dict. Invalid temperature values become None
    and the sensor is listed in 'bad_sensors' so the UI can say so.
    """
    if not isinstance(r, dict):
        return {}
    out, bad = {}, []
    for tag, key in SENSOR_KEYS.items():
        v = num(r.get(key))
        if v is not None and not valid_temp(v):
            bad.append(tag)
            v = None
        out[tag.lower()] = v
    flow = num(r.get("flow_lpm"))
    if "pump_status" in r and r.get("pump_status") is not None:
        pump = bool(r["pump_status"])
    else:
        pump = None if flow is None else flow >= PUMP_FLOW_THRESHOLD
    w = r.get("weather") or {}
    out.update({
        "epoch": to_epoch(_first(r.get("ts"), r.get("timestamp"))),
        "flow": flow,
        "volume": num(r.get("volume_l")),
        "pulses": num(r.get("pulses")),
        "pump": pump,
        "pump_inferred": not ("pump_status" in r and r.get("pump_status") is not None),
        "mode": r.get("mode"),
        "ambient": num(w.get("ambient_temp")),
        "irradiance": num(w.get("irradiance")),
        "weather_epoch": to_epoch(w.get("timestamp")),
        "bad_sensors": bad,
    })
    return out


def read_recent(db, minutes=60, max_records=1500):
    """
    The last `minutes` of history measured back from the NEWEST record (not from now), so the
    live chart still shows something when the installation is offline. Returns a list of parsed
    records, oldest first.
    """
    if not db:
        return []
    try:
        snap = db.reference(HISTORY_PATH).order_by_child("ts").limit_to_last(max_records).get()
    except Exception:
        return []
    if not snap:
        return []
    recs = [parse_record(r) for r in (snap.values() if isinstance(snap, dict) else snap) if isinstance(r, dict)]
    recs = [r for r in recs if r.get("epoch") is not None]
    recs.sort(key=lambda r: r["epoch"])
    if not recs:
        return []
    cutoff = recs[-1]["epoch"] - minutes * 60
    return [r for r in recs if r["epoch"] >= cutoff]


# ---------------------------------------------------------
# History -> DataFrame (shared by Historical Trends and Fault Logs)
# ---------------------------------------------------------
def finish_raw(df):
    """UTC datetime index, numeric columns, sorted, no duplicates."""
    for c in RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    t = pd.to_numeric(df["t"], errors="coerce")
    t = t.where(t < 1e11, t / 1000.0)  # accept ms timestamps too
    df.index = pd.to_datetime(t, unit="s", utc=True)
    df = df[df.index.notna()].drop(columns="t")
    return df[~df.index.duplicated()].sort_index()


def records_to_frame(snap):
    """Firebase snapshot (dict of records or list) -> raw frame. Works for site history and sim samples."""
    if not snap:
        return pd.DataFrame()
    records = snap.values() if isinstance(snap, dict) else [r for r in snap if r]
    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        w = r.get("weather") or {}
        pump = _first(r.get("pump"), r.get("pump_status"))
        heater_on = r.get("heater_on")
        rows.append({
            "t": _first(r.get("ts"), r.get("timestamp")),
            "S1": r.get("s1_collector"),
            "S2": r.get("s2_tank_mid"),
            "S3": r.get("s3_tank_top"),
            "S4": r.get("s4_tank_bottom"),
            "Ambient": _first(r.get("ambient_temp"), w.get("ambient_temp")),
            "Irradiance": _first(r.get("irradiance"), w.get("irradiance")),
            "Flow": r.get("flow_lpm"),
            "Pump": None if pump is None else float(bool(pump)),
            "Heater_W": r.get("heater_power_w"),
            "Heater_on": None if heater_on is None else float(bool(heater_on)),
        })
    if not rows:
        return pd.DataFrame()
    return finish_raw(pd.DataFrame(rows))


@st.cache_data(ttl=120, show_spinner="Loading history from Firebase…")
def load_history(_db, start_epoch, end_epoch):
    snap = (
        _db.reference(HISTORY_PATH)
        .order_by_child("ts")
        .start_at(start_epoch)
        .end_at(end_epoch)
        .get()
    )
    return records_to_frame(snap)


def sanitise(df):
    """Blank out impossible temperatures (e.g. -127 from a disconnected DS18B20) so they don't wreck charts."""
    df = df.copy()
    for c in ("S1", "S2", "S3", "S4"):
        df[c] = df[c].where((df[c] >= -55) & (df[c] <= 125))
    return df


# ---------------------------------------------------------
# Weather (written by weather_to_firebase.py, NOT measured on site)
# ---------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def read_weather_latest(_db):
    """{'ambient', 'irradiance', 'epoch'} from site/weather/latest, or {} when unavailable."""
    if not _db:
        return {}
    try:
        w = _db.reference(WEATHER_LATEST_PATH).get()
    except Exception:
        return {}
    if not isinstance(w, dict):
        return {}
    return {"ambient": num(w.get("ambient_temp")), "irradiance": num(w.get("irradiance")),
            "epoch": to_epoch(w.get("ts"))}


@st.cache_data(ttl=300, show_spinner=False)
def load_weather(_db, start_epoch, end_epoch):
    """Weather samples around [start, end] as a UTC-indexed frame with Ambient / Irradiance columns."""
    try:
        snap = (_db.reference(WEATHER_HISTORY_PATH).order_by_child("ts")
                .start_at(start_epoch - 7200).end_at(end_epoch + 7200).get())
    except Exception:
        return pd.DataFrame()
    if not snap:
        return pd.DataFrame()
    rows = [{"t": to_epoch(r.get("ts")), "Ambient": num(r.get("ambient_temp")), "Irradiance": num(r.get("irradiance"))}
            for r in (snap.values() if isinstance(snap, dict) else snap) if isinstance(r, dict)]
    rows = [r for r in rows if r["t"] is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("t").drop_duplicates("t")
    df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    return df


def _secs(index):
    return (index - pd.Timestamp(0, tz="UTC")).total_seconds().to_numpy()


def attach_weather(raw, wx, max_gap_s=5400):
    """
    Fill raw['Ambient'] / raw['Irradiance'] from weather samples by linear interpolation in time.
    Readings further than max_gap_s from any weather sample stay empty (no invented weather).
    """
    if raw is None or raw.empty or wx is None or wx.empty:
        return raw
    raw = raw.copy()
    rt, wt = _secs(raw.index), _secs(wx.index)
    pos = np.clip(np.searchsorted(wt, rt), 1, len(wt) - 1) if len(wt) > 1 else np.zeros(len(rt), dtype=int)
    nearest = np.minimum(np.abs(rt - wt[np.clip(pos - 1, 0, len(wt) - 1)]), np.abs(rt - wt[pos]))
    ok = nearest <= max_gap_s
    for col in ("Ambient", "Irradiance"):
        vals = wx[col].to_numpy(dtype=float)
        good = ~np.isnan(vals)
        if good.sum() == 0:
            continue
        interp = np.interp(rt, wt[good], vals[good])
        raw[col] = np.where(ok, interp, np.nan)
    return raw


# ---------------------------------------------------------
# Simulation runs (saved by the Digital Twin page)
# ---------------------------------------------------------
def save_sim_run(db, run_id, meta, samples):
    """Write a (possibly partial) simulation run. Safe to call repeatedly with the same run_id."""
    if not db or not samples:
        return False
    try:
        meta = dict(meta, samples=len(samples), updated=datetime.now().timestamp())
        db.reference(f"{SIM_RUNS_PATH}/{run_id}").set({"meta": meta, "samples": samples})
        db.reference(f"{SIM_INDEX_PATH}/{run_id}").set(meta)
        return True
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def list_sim_runs(_db):
    """[{id, label, started, samples}, ...] newest first."""
    try:
        idx = _db.reference(SIM_INDEX_PATH).get() or {}
    except Exception:
        return []
    runs = [dict(m, id=k) for k, m in idx.items() if isinstance(m, dict)]
    runs.sort(key=lambda m: m.get("started") or 0, reverse=True)
    return runs


@st.cache_data(ttl=60, show_spinner="Loading simulation run…")
def load_sim_run(_db, run_id):
    snap = _db.reference(f"{SIM_RUNS_PATH}/{run_id}/samples").get()
    return records_to_frame(snap)


# ---------------------------------------------------------
# Anomaly detection on real telemetry (for the Fault Logs page)
# ---------------------------------------------------------
_SEV_ICON = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}


def _runs(mask, idx, min_s=0):
    """Contiguous True runs in `mask` lasting at least min_s -> [(start_ts, end_ts, ongoing)]."""
    m = mask.fillna(False).to_numpy()
    out, start = [], None
    for i, flag in enumerate(m):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(m) - 1):
            end = i if flag else i - 1
            if (idx[end] - idx[start]).total_seconds() >= min_s:
                out.append((idx[start], idx[end], flag and i == len(m) - 1))
            start = None
    return out


def detect_anomalies(raw):
    """
    Rule-based anomalies from raw site history (UTC index). These are DERIVED from telemetry;
    the controller does not log them, so treat them as flags to investigate, not verdicts.
    Returns a DataFrame, newest first.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.sort_index()
    idx = df.index
    events = []

    def add(start, end, kind, sev, detail, ongoing=False):
        events.append({
            "Start": start, "End": end, "Type": kind,
            "Severity": f"{_SEV_ICON[sev]} {sev}",
            "Status": "Ongoing" if ongoing else "Ended",
            "Duration (min)": round((end - start).total_seconds() / 60, 1),
            "Detail": detail,
        })

    # 1. Data gaps (node off / offline). Compared with the node's own logging interval.
    step = idx.to_series().diff().dt.total_seconds()
    median = step.median()
    gap_limit = max(300.0, 5 * median) if pd.notna(median) else 300.0
    for i in np.where(step.to_numpy() > gap_limit)[0]:
        add(idx[i - 1], idx[i], "Data gap", "WARNING",
            f"No readings for {step.iloc[i] / 60:.0f} min (installation off, offline or logging stopped).")

    # 2. Invalid sensor values (-127 = DS18B20 disconnected)
    for tag in ("S1", "S2", "S3", "S4"):
        bad = df[tag].notna() & ((df[tag] < -55) | (df[tag] > 125))
        for s, e, on in _runs(bad, idx):
            add(s, e, f"Sensor {tag} invalid", "CRITICAL",
                f"{tag} reported {df.loc[s, tag]:.0f} °C (wiring / connector / sensor fault).", on)

    clean = sanitise(df)

    # 3. Overheat
    hot = (clean["S1"] >= OVERHEAT_C) | (clean["S3"] >= OVERHEAT_C)
    for s, e, on in _runs(hot, idx):
        peak = clean.loc[s:e, ["S1", "S3"]].max().max()
        add(s, e, "Overheat", "CRITICAL", f"S1 or S3 at/above {OVERHEAT_C:.0f} °C (peak {peak:.1f} °C).", on)

    # 4. Collector hot, no circulation
    flow = clean["Flow"]
    dt = clean["S1"] - clean["S4"]
    no_circ = (dt >= HOT_COLLECTOR_DT) & (flow < PUMP_FLOW_THRESHOLD)
    for s, e, on in _runs(no_circ, idx, SUSTAIN_S):
        add(s, e, "Hot collector, no flow", "WARNING",
            f"S1 − S4 ≥ {HOT_COLLECTOR_DT:.0f} K with flow < {PUMP_FLOW_THRESHOLD} L/min "
            "(pump off, air lock, closed valve or dry run).", on)

    # 5. Circulating while the collector is colder than the tank (heat being lost)
    losing = (flow >= PUMP_FLOW_THRESHOLD) & (dt < 0)
    for s, e, on in _runs(losing, idx, SUSTAIN_S):
        add(s, e, "Circulating, collector colder than tank", "INFO",
            "Flow with S1 < S4: the loop is moving heat from the tank to the collector (check pump control / ΔT thresholds).", on)

    if not events:
        return pd.DataFrame()
    out = pd.DataFrame(events).sort_values("Start", ascending=False).reset_index(drop=True)
    for c in ("Start", "End"):
        out[c] = out[c].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out
