import time

import streamlit as st

from utils import site_data as sd


def _age(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h {(seconds % 3600) // 60} min ago"
    return f"{seconds // 86400} d {(seconds % 86400) // 3600} h ago"


def _run_test():
    try:
        db = sd.get_db()
    except Exception as e:
        st.error(f"🔴 Connection failed: {type(e).__name__}: {e}")
        return
    if not db:
        reason = sd.last_error()
        st.error("🔴 Firebase is not connected" + (f": {reason}" if reason else "."))
        return
    st.success("🟢 Firebase connection OK")

    try:
        raw, err = sd.read_latest(db)
    except Exception as e:
        st.error(f"Couldn't read `site/latest`: {type(e).__name__}: {e}")
        return
    if raw is None:
        st.warning("Connected, but no reading found at `site/latest`" + (f" ({err})" if err else "") + ".")
        return

    epoch = sd.parse_record(raw)["epoch"]
    if epoch:
        st.write(f"**Last reading from the installation:** {sd.fmt_local(epoch)} ({_age(time.time() - epoch)})")
    else:
        st.write("A reading exists at `site/latest`, but it has no timestamp.")

    try:
        n = len(sd.read_recent(db, 60))
        st.write(f"**History log:** {n} readings returned for the last 60 min window.")
    except Exception as e:
        st.warning(f"Couldn't read `site/history`: {type(e).__name__}: {e}")


def show():
    col_nav, col_title = st.columns([1, 5])
    with col_nav:
        if st.button("⬅ Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    st.title("⚙️ Settings")
    st.caption("Connection status and diagnostics.")
    st.divider()

    st.subheader("🔥 Firebase connection")
    st.write("Checks that the dashboard can reach the database and that the installation is still reporting.")
    if st.button("Run connection test", type="primary"):
        _run_test()