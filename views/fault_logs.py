"""
Fault Logs: every fault event the HIL fault engine has recorded (hil/faults in Firebase).

Resolved faults are never removed: they stay as history with their recovery time.
All rows here are HIL TEST events (injected or physically detected during a test).
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from hil import dashboard as hp

TZ = "Africa/Johannesburg"


def _local(epoch):
    if epoch is None:
        return None
    return pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _mmss(seconds):
    if seconds is None:
        return None
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_table(events: dict, sms: dict) -> pd.DataFrame:
    """One row per event, newest first. Pure function (no Streamlit), easy to test."""
    rows = []
    for ev in (events or {}).values():
        if not isinstance(ev, dict):
            continue
        sms_ids = [s.get("sms_id") for s in (ev.get("sms") or []) if isinstance(s, dict)]
        sms_now = []
        for s in (ev.get("sms") or []):
            if not isinstance(s, dict):
                continue
            current = (sms or {}).get(s.get("sms_id"), {})
            sms_now.append(f"{s.get('kind')}: {current.get('status', s.get('status'))}")
        actions = [a.get("action") for a in (ev.get("mitigation_actions") or []) if isinstance(a, dict)]
        det_epoch, det_s = ev.get("detection_epoch"), ev.get("detection_s")
        started = _local(det_epoch - det_s) if det_epoch is not None and det_s is not None else None
        rows.append({
            "Event ID": ev.get("event_id"),
            "Fault": ev.get("fault_type"),
            "Subject": ev.get("subject"),
            "Severity (peak)": f"{hp.SEV_ICON.get(ev.get('peak_severity'), '')} {ev.get('peak_severity')}",
            "Source": ev.get("source"),
            "Status": ev.get("status"),
            "Run started": started,
            "Detected (test mm:ss)": _mmss(det_s),
            "Recovered (test mm:ss)": _mmss(ev.get("recovery_s")),
            "Detection delay (s)": ev.get("detection_delay_s"),
            "Time to recover (s)": ev.get("time_to_recover_s"),
            "Mitigation actions": ", ".join(dict.fromkeys(actions)) or "—",
            "SMS": "; ".join(sms_now) or "—",
            "_sort": ev.get("detection_epoch") or 0,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return df


def show():
    head1, head2 = st.columns([6, 1])
    with head1:
        st.title("⚠ Fault Logs")
        st.caption("Every fault the system has detected during HIL tests. Resolved faults stay here as history.")
    with head2:
        if st.button("⬅ Home"):
            st.session_state.page = "Home"
            st.rerun()

    events, err = hp.read_node("hil/faults")
    sms, _ = hp.read_node("hil/sms")
    if err and events is None:
        st.error(f"Could not read the fault records: {err}")
        return
    if not events:
        st.info("No faults recorded yet. Run a test from the Digital Twin page (HIL mode).")
        return

    df = build_table(events, sms)
    f1, f2, f3 = st.columns(3)
    status = f1.multiselect("Status", sorted(df["Status"].dropna().unique()), default=list(df["Status"].dropna().unique()))
    kinds = f2.multiselect("Fault type", sorted(df["Fault"].dropna().unique()), default=list(df["Fault"].dropna().unique()))
    sources = f3.multiselect("Source", sorted(df["Source"].dropna().unique()), default=list(df["Source"].dropna().unique()))
    view = df[df["Status"].isin(status) & df["Fault"].isin(kinds) & df["Source"].isin(sources)]

    k1, k2, k3 = st.columns(3)
    k1.metric("Events shown", len(view))
    k2.metric("Active", int((view["Status"] != "RESOLVED").sum()))
    k3.metric("Resolved", int((view["Status"] == "RESOLVED").sum()))

    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("⬇ Download CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="stmacs_fault_log.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Event details")
    ids = list(view["Event ID"]) if len(view) else []
    if ids:
        chosen = st.selectbox("Event", ids, key="fl_event")
        ev = events.get(chosen) or {}
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Measured at detection**")
            st.json(ev.get("measured_at_detection") or {})
            st.markdown("**Severity history** (test seconds, severity)")
            st.json(ev.get("severity_history") or [])
        with d2:
            st.markdown("**Measured at peak**")
            st.json(ev.get("measured_at_peak") or {})
            st.markdown("**Mitigation actions**")
            acts = ev.get("mitigation_actions") or []
            if acts:
                st.dataframe([{"Test s": a.get("t"), "Action": a.get("action"), "Domain": a.get("domain"),
                               "Detail": a.get("detail")} for a in acts], hide_index=True, use_container_width=True)
            else:
                st.caption("none")
        if ev.get("injections"):
            st.caption(f"Injected faults active at detection: {', '.join(ev['injections'])}")
    st.caption("Thresholds behind the severity levels are EXPERIMENTAL HIL test settings, not validated HOTSET limits.")
