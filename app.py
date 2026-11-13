import streamlit as st

from utils.authentication import login, check_login, logout
from hil import dashboard as hil_dashboard

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
    # DEFAULT PAGE (LANDS ON HOME)
    # ------------------------------------------------------

    if "page" not in st.session_state:
        st.session_state.page = "Home"


    # ------------------------------------------------------
    # PAGE ROUTER
    # ------------------------------------------------------

    page = st.session_state.page

    # HIL fault pop-ups: shown on every page
    hil_dashboard.notifier()


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