import streamlit as st

from utils.authentication import login, check_login, logout

from views import (
    home,
    live_monitoring,
    digital_twin,
    historical_trends,
    predictive_control,
    fault_logs,
    settings,
    about
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="STMaCS Dashboard",
    page_icon="☀️",
    layout="wide"
)


# ==========================================================
# LOGIN PAGE
# ==========================================================

def login_page():

    st.title("☀️ Smart Solar Thermal Monitoring and Control System")

    st.write("Please login to access the dashboard.")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )


    if st.button("Login"):

        if login(username, password):

            st.success("Login successful")

            st.rerun()

        else:

            st.error("Incorrect username or password")



# ==========================================================
# PAGE ROUTER
# ==========================================================

def dashboard():



    # ------------------------------------------------------
    # DEFAULT PAGE
    # ------------------------------------------------------

    if "page" not in st.session_state:
        st.session_state.page = "Home"


    # ------------------------------------------------------
    # SIDEBAR NAVIGATION
    # ------------------------------------------------------

    with st.sidebar:

        st.title("☀️ STMaCS")

        st.caption("Smart Solar Thermal Monitoring and Control System")

        st.divider()


        # Main pages
        st.subheader("Dashboard")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

        if st.button("📡 Live Monitoring", use_container_width=True):
            st.session_state.page = "Live Monitoring"
            st.rerun()

        if st.button("📈 Historical Trends", use_container_width=True):
            st.session_state.page = "Historical Trends"
            st.rerun()

        if st.button("🧩 Digital Twin", use_container_width=True):
            st.session_state.page = "Digital Twin"
            st.rerun()

        if st.button("🤖 Predictive Control", use_container_width=True):
            st.session_state.page = "Predictive Control"
            st.rerun()

        if st.button("⚠️ Fault Logs", use_container_width=True):
            st.session_state.page = "Fault Logs"
            st.rerun()


        st.divider()


        # Secondary pages
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "About Project"
            st.rerun()

        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "Settings"
            st.rerun()


        st.divider()


        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()


    # ------------------------------------------------------
    # PAGE ROUTER
    # ------------------------------------------------------

    page = st.session_state.page


    if page == "Home":
        home.show()

    elif page == "Live Monitoring":
        live_monitoring.show()

    elif page == "Digital Twin":
        digital_twin.show()

    elif page == "Historical Trends":
        historical_trends.show()

    elif page == "Predictive Control":
        predictive_control.show()

    elif page == "Fault Logs":
        fault_logs.show()

    elif page == "Settings":
        settings.show()

    elif page == "About Project":
        about.show()



# ==========================================================
# APPLICATION START
# ==========================================================

if check_login():

    dashboard()

else:

    login_page()