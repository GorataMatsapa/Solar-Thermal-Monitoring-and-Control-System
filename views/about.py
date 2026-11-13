import streamlit as st


def show():
    col_nav, col_title = st.columns([1, 5])
    with col_nav:
        if st.button("⬅ Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    st.title("ℹ️ About STMaCS")
    st.caption("Solar Thermal Monitoring and Control System")
    st.divider()

    st.info(
        "STMaCS monitors a solar thermal hot-water installation (HOTSET 300 L tank). "
        "An ESP32 node reads temperature and flow sensors and sends the data over cellular to a cloud "
        "database. This dashboard shows it live and as history, and includes a digital twin for simulation. "
        "Predictive control is being developed on top of it."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What it does today")
        st.markdown(
            """
            * **Live Monitoring:** latest sensor readings, pump and flow status, and on-screen alerts
              (overheat, invalid sensor, no flow). Shows the last reading time if the site goes offline.
            * **Historical Analytics:** temperature, flow, pump runtime and estimated energy over a chosen period.
            * **Digital Twin:** simulate the system and save runs for review in Historical Analytics.
            """
        )
        st.subheader("In development")
        st.markdown("* **Predictive control:** adjusting heating ahead of time.")

    with col2:
        st.subheader("How it works")
        st.markdown(
            """
            * **Edge node:** ESP32-S3 with a SIM7670G cellular module
            * **Cloud:** Firebase Realtime Database
            * **Dashboard:** Streamlit
            """
        )

    st.divider()

    st.subheader("📡 Sensors")
    st.table([
        {"Sensor": "S1", "Measures": "Collector line temperature", "Type": "DS18B20"},
        {"Sensor": "S2", "Measures": "Tank middle temperature", "Type": "DS18B20"},
        {"Sensor": "S3", "Measures": "Tank top temperature (hot water out)", "Type": "DS18B20"},
        {"Sensor": "S4", "Measures": "Tank bottom temperature (collector return)", "Type": "DS18B20"},
        {"Sensor": "Flow", "Measures": "Collector loop flow and circulated volume", "Type": "Pulse flow meter"},
    ])
    st.caption("Outdoor temperature and irradiance come from a weather service, not from on-site sensors.")

    st.divider()
    st.caption("STMaCS Build Version: v1.0.0 | University of Botswana | Clean Energy Research Center | AEE INTEC")