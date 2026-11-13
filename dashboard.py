"""
Dashboard side of HIL (lives in the hil/ package, so views/ keeps only your own pages).
Includes the fault pop-up notifier `notifier()` that app.py calls on every page.

Shared HIL (hardware-in-the-loop) widgets for the dashboard.

Reads the "latest value" nodes the HIL bridge writes in Firebase:
    hil/live       everything the bridge knows right now (sensors, pump, flow, faults, SMS ...)
    hil/physical   what the ESP32 itself reports (real sensor, relay states)
and writes hil/control when you press Start / Stop.

Every value is labelled PHYSICAL (a real sensor on the ESP32), SIMULATED (digital twin) or
INJECTED (a test fault applied on purpose). HIL data is never mixed with live_telemetry.
"""

import time

import streamlit as st

SCENARIOS = {
    "T1": "Test 1 - Normal operation",
    "T2": "Test 2 - Pump failure / no circulation",
    "T3": "Test 3 - Low flow",
    "T4A": "Test 4, Run A - Stagnation, NO mitigation",
    "T4B": "Test 4, Run B - Stagnation, WITH simulated mitigation",
    "T5": "Test 5 - Sensor failure",
    "T6": "Test 6 - Fault recovery",
}
SEV_ICON = {"warning": "🟡", "major": "🟠", "critical": "🔴"}
SRC_LABEL = {"physical": "🟢 PHYSICAL", "simulated": "🟣 SIMULATED", "injected": "🟠 INJECTED"}
BRIDGE_OFFLINE_AFTER_S = 20


def get_db():
    """The firebase_admin db module (session_state.db if set, otherwise initialise it here)."""
    db = st.session_state.get("db")
    if db:
        return db
    try:
        import utils.firebase  # noqa: F401  (initialises firebase_admin with your service account)
        from firebase_admin import db as firebase_db
        return firebase_db
    except Exception:
        return None


def read_node(path):
    """(value, error). Never raises, so a dropped connection cannot crash the page."""
    db = get_db()
    if db is None:
        return None, "Firebase is not connected."
    try:
        return db.reference(path).get(), None
    except Exception as exc:                      # network, permissions ...
        return None, str(exc)


def send_command(scenario_cmd, s1_source="simulated", flow_source="simulated"):
    db = get_db()
    if db is None:
        return False, "Firebase is not connected."
    try:
        db.reference("hil/control").update({
            "cmd_id": int(time.time() * 1000), "scenario_cmd": scenario_cmd,
            "s1_source": s1_source, "flow_source": flow_source,
        })
        return True, ""
    except Exception as exc:
        return False, str(exc)


def fmt(v, unit="", decimals=1):
    return "—" if v is None else f"{v:.{decimals}f}{(' ' + unit) if unit else ''}"


def onoff(v):
    return "—" if v is None else ("ON" if v else "OFF")


def bridge_status(live):
    """('online'|'offline'|'none', age_s)"""
    if not live or live.get("updated_epoch") is None:
        return "none", None
    age = time.time() - float(live["updated_epoch"])
    return ("online" if age <= BRIDGE_OFFLINE_AFTER_S else "offline"), age


# ---------------------------------------------------------------------------------------
# compact strip (used on the Live Monitoring page)
# ---------------------------------------------------------------------------------------
def _strip_body():
    live, err = read_node("hil/live")
    state, age = bridge_status(live)
    if state == "none":
        st.caption("🧪 HIL test: no test bridge data yet (start `run_hil_live.py`, then open Digital Twin, HIL mode).")
        return
    hw = live.get("hardware") or {}
    pump = live.get("pump") or {}
    flow = live.get("flow") or {}
    faults = (live.get("faults") or {}).get("open") or []
    sms = (live.get("sms") or {}).get("latest") or {}
    mit = live.get("mitigation") or {}

    st.markdown("##### 🧪 HIL test status  ·  *test data, not HOTSET telemetry*")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"Bridge: {'🟢 online' if state == 'online' else '🔴 offline'}")
    c2.markdown(f"ESP32: {'🟢 online' if hw.get('online') else '🔴 offline'}")
    c3.markdown(f"Test: **{live.get('scenario', '—')}** ({live.get('state', '—')})")
    c4.markdown(f"Time: {fmt(live.get('t_s'), 's', 0)}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pump command / actual", f"{onoff(pump.get('command'))} / {onoff(pump.get('actual'))}")
    m2.metric(f"Flow ({flow.get('source', '—')})", fmt(flow.get('measured_lpm'), 'L/min', 2))
    if faults:
        worst = max(faults, key=lambda f: ["warning", "major", "critical"].index(f.get("severity", "warning")))
        m3.metric("Active fault", f"{SEV_ICON.get(worst.get('severity'), '')} {worst.get('fault_type')}", worst.get("severity"))
    else:
        m3.metric("Active fault", "none")
    m4.metric("SMS", sms.get("status", "—"))
    if mit.get("protection_latched"):
        st.warning("Simulated protection is ACTIVE (simulation only: shading + simulated valve).")


_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _fragment is not None:
    _strip_body = _fragment(run_every=5)(_strip_body)


def status_strip():
    _strip_body()


# ---------------------------------------------------------------------------------------
# full live panel (HIL Test page)
# ---------------------------------------------------------------------------------------
def render_live():
    live, err = read_node("hil/live")
    phys, _ = read_node("hil/physical")
    if err and live is None:
        st.error(f"Could not read Firebase: {err}")
        return
    state, age = bridge_status(live)
    if state == "none":
        st.info("No HIL bridge data yet. On your computer run `python run_hil_live.py`, then press Start.")
        return
    if state == "offline":
        st.error(f"The HIL bridge stopped updating {age:.0f} s ago. Values below are the last known ones.")

    hw = live.get("hardware") or {}
    sensors = live.get("sensors") or {}
    pump = live.get("pump") or {}
    flow = live.get("flow") or {}
    twin = live.get("twin") or {}
    faults = (live.get("faults") or {}).get("open") or []
    mit = live.get("mitigation") or {}

    # ---- header ----
    h1, h2, h3 = st.columns([2, 2, 3])
    h1.markdown(f"**{live.get('scenario_name', 'No test selected')}**  \nState: `{live.get('state', '—')}`")
    h2.markdown(f"Bridge: {'🟢 online' if state == 'online' else '🔴 offline'}  \n"
                f"ESP32: {'🟢 online' if hw.get('online') else '🔴 offline'}"
                + (f" ({hw.get('age_s')} s)" if hw.get("age_s") is not None else ""))
    dur, t = live.get("duration_s"), live.get("t_s")
    if dur and t is not None:
        h3.progress(min(max(t / dur, 0.0), 1.0), text=f"Test time {t:.0f} s of {dur:.0f} s  ({live.get('time_scale', 1):g}x speed)")
    if hw.get("s1_requested") == "physical" and hw.get("s1_effective") != "physical":
        st.warning("S1 was set to the REAL sensor but the ESP32 is offline, so the simulated S1 is being used.")

    st.markdown("---")

    # ---- hardware (real) ----
    st.subheader("🔌 Real hardware (ESP32)")
    pr = hw.get("physical_reading") or phys or {}
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("DS18B20 (real S1)", fmt(pr.get("s1_c"), "°C", 1) if pr.get("s1_ok", True) else "no reading",
              help="Warm the probe in warm water, or unplug it for a real sensor fault.")
    r2.metric("Flow sensor (real)", fmt(pr.get("flow_lpm"), "L/min", 2),
              help="Only used by the tests when 'flow source' is set to PHYSICAL.")
    r3.metric("Fan relay (actual)", onoff(pr.get("fan_on")))
    r4.metric("Valve relay (bench valve)", onoff(pr.get("valve_on")))

    # ---- pump / flow ----
    st.subheader("⚙ Pump and flow")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Pump COMMAND", onoff(pump.get("command")))
    p2.metric("Pump ACTUAL (fan)", onoff(pump.get("actual")))
    p3.metric("Flow", fmt(flow.get("measured_lpm"), "L/min", 2), flow.get("source"), delta_color="off")
    p4.metric("Expected flow", fmt(flow.get("expected_lpm"), "L/min", 1))
    if pump.get("command") and not pump.get("actual"):
        st.error("Pump command is ON but the actual pump (fan) is OFF.")
    if pump.get("fallback_off"):
        st.warning("Pump fallback: S1 or S4 invalid, pump commanded OFF.")

    # ---- sensors ----
    st.subheader("🌡 Sensors (value the controllers are allowed to use)")
    rows = []
    for sid in ("S1", "S2", "S3", "S4"):
        s = sensors.get(sid) or {}
        rows.append({"Sensor": sid, "Value (°C)": s.get("value"),
                     "Source": SRC_LABEL.get(s.get("source"), s.get("source")),
                     "Reliable": "yes" if s.get("reliable") else "NO - rejected"})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Twin collector (truth)", fmt(twin.get("collector_c"), "°C"))
    t2.metric("Twin tank top", fmt(twin.get("tank_top_c"), "°C"))
    t3.metric("Solar input", fmt(twin.get("irradiance_wm2"), "W/m²", 0),
              f"shading x{twin.get('shading_factor', 1)}" if twin.get("shading_factor", 1) != 1 else None, delta_color="off")
    t4.metric("Simulated valve", "OPEN" if (live.get("valve_SIM") or {}).get("open") else "closed",
              help="SIMULATION-ONLY heat-dump valve; shown on the bench valve. Not connected to any pipework.")
    if live.get("injections_active"):
        st.caption(f"🟠 Test faults currently injected: {live['injections_active']}")

    # ---- faults ----
    st.subheader("⚠ Faults, reaction and notification")
    if faults:
        st.dataframe([{"Severity": f"{SEV_ICON.get(f.get('severity'), '')} {f.get('severity')}",
                       "Fault": f.get("fault_type"), "Subject": f.get("subject"),
                       "Source": f.get("source"), "Status": f.get("status"),
                       "Detected at (test s)": f.get("detection_s")} for f in faults],
                     hide_index=True, use_container_width=True)
    else:
        st.success("No active fault.")
    if mit.get("protection_latched"):
        st.warning("Simulated protection is ACTIVE (simulation only).")
    acts = mit.get("recent_actions") or []
    if acts:
        st.caption("Recent actions (PHYSICAL = would go to hardware, SIMULATION = digital twin only):")
        st.dataframe([{"Test s": a.get("t"), "Action": a.get("action"), "Domain": a.get("domain"),
                       "Detail": a.get("detail")} for a in reversed(acts)], hide_index=True, use_container_width=True)
    sms = (live.get("sms") or {})
    if sms.get("latest"):
        st.caption(f"SMS requests: {sms.get('requests')}  ·  latest status: **{sms['latest'].get('status')}** "
                   f"({sms['latest'].get('detail', '')})")


# ---------------------------------------------------------------------------------------
# Digital Twin "HIL mode" helpers
# ---------------------------------------------------------------------------------------
def twin_state_from_live(live):
    """State dict for the Digital Twin SVG from hil/live (None if there is no running test)."""
    tv = (live or {}).get("twin_view")
    if not tv:
        return None
    state = dict(tv)
    state["mode"] = "HIL"
    state["simulation_time"] = round((live.get("t_s") or 0) / 60.0, 1)   # test minutes
    state["system_status"] = "HIL TEST: " + str(live.get("scenario_name", ""))
    return state


def render_controls():
    """Start / Stop controls (used inside the Digital Twin page in HIL mode)."""
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            sid = st.selectbox("Test", list(SCENARIOS), format_func=lambda k: SCENARIOS[k], key="hil_sid")
        with c2:
            s1_src = st.radio("S1 (collector) sensor", ["simulated", "physical"], horizontal=True, key="hil_s1src",
                              help="PHYSICAL = the real DS18B20 on the ESP32.")
        with c3:
            flow_src = st.radio("Flow sensor", ["simulated", "physical"], horizontal=True, key="hil_flowsrc",
                                help="PHYSICAL = the real YF-S201 (pour water through it).")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("▶ Start test", type="primary", use_container_width=True, key="hil_start"):
            ok, msg = send_command(sid, s1_src, flow_src)
            (st.success if ok else st.error)(f"{SCENARIOS[sid]} requested." if ok else f"Could not send: {msg}")
        if b2.button("⏹ Stop", use_container_width=True, key="hil_stop"):
            ok, msg = send_command("stop", s1_src, flow_src)
            (st.info if ok else st.error)("Stop requested." if ok else f"Could not send: {msg}")


# ---------------------------------------------------------------------------------------
# Fault pop-ups on EVERY page (called once from app.py)
# ---------------------------------------------------------------------------------------

ALERT_CHECK_EVERY_S = 3
_ALERT_RANK = {"warning": 0, "major": 1, "critical": 2}


def _sev_text(f):
    return f"{SEV_ICON.get(f.get('severity'), '')} {f.get('severity', '').upper()}"


def _sms_popups(live, ss):
    """Toast the crew-notification result. 'Notified' is only shown when the service CONFIRMED it."""
    ss.setdefault("hil_sms_done", set())
    primed = ss.get("hil_sms_primed", False)
    for r in ((live.get("sms") or {}).get("recent") or []):
        sid, status = r.get("sms_id"), r.get("status")
        if not sid or sid in ss["hil_sms_done"] or status in ("pending", "queued", "sending", None):
            continue
        ss["hil_sms_done"].add(sid)
        if not primed:
            continue                                  # do not replay old notifications when the page opens
        if status == "simulated":
            st.toast("Maintenance crew alert triggered (SMS module not connected for this run, no message sent)", icon="📲")
            continue
        what = "SMS" if r.get("channel") == "sms" else "push notification (not an SMS)"
        if status in ("sent", "delivered"):
            st.toast(f"Maintenance crew notified by {what}: {r.get('kind', '')}", icon="📲")
        elif status == "failed":
            st.toast(f"Maintenance notification FAILED: {r.get('detail', '')}", icon="❌")
        elif status == "unavailable":
            st.toast("Maintenance crew could NOT be notified: no SMS channel is connected", icon="⚠️")
    ss["hil_sms_primed"] = True


def _alerts_body():
    ss = st.session_state
    ss.setdefault("hil_alert_seen", {})      # event_id -> last severity/status we announced
    ss.setdefault("hil_alert_acked", set())  # (event_id, severity) acknowledged
    ss.setdefault("hil_alert_primed", False)

    live, _ = read_node("hil/live")
    bstate, _age = bridge_status(live)
    if bstate == "none":
        return
    open_faults = [f for f in ((live.get("faults") or {}).get("open") or []) if isinstance(f, dict)]
    current = {f["event_id"]: f for f in open_faults if f.get("event_id")}
    seen = ss["hil_alert_seen"]

    for eid, f in current.items():
        key = f"{f.get('severity')}|{f.get('status')}"
        if seen.get(eid) != key:
            first = eid not in seen
            if ss["hil_alert_primed"] or f.get("severity") == "critical":
                label = "New fault" if first else "Fault update"
                st.toast(f"{label}: {f.get('fault_type')} ({f.get('subject')}) - {_sev_text(f)} [HIL test]",
                         icon="🚨" if f.get("severity") == "critical" else "⚠️")
            seen[eid] = key
    for eid in [e for e in seen if e not in current]:
        if ss["hil_alert_primed"]:
            st.toast(f"Fault resolved: {eid.split('_')[-1] if '_' in eid else eid} [HIL test]", icon="✅")
        del seen[eid]
    ss["hil_alert_primed"] = True

    _sms_popups(live, ss)

    for eid, f in current.items():
        if f.get("severity") != "critical" or (eid, f.get("severity")) in ss["hil_alert_acked"]:
            continue
        c1, c2 = st.columns([8, 1])
        c1.error(f"🔴 CRITICAL FAULT (HIL test): **{f.get('fault_type')}** on {f.get('subject')} · "
                 f"status {f.get('status')} · detected at test second {f.get('detection_s')}. "
                 "Test data, not a real HOTSET alarm.")
        if c2.button("Acknowledge", key=f"ack_{eid}"):
            ss["hil_alert_acked"].add((eid, f.get("severity")))
            st.rerun(scope="fragment")


_alert_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
notifier = _alert_fragment(run_every=ALERT_CHECK_EVERY_S)(_alerts_body) if _alert_fragment is not None else _alerts_body
