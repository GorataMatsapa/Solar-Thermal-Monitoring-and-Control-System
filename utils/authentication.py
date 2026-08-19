import streamlit as st


# Dummy credentials (temporary)
USERNAME = "username"
PASSWORD = "password"


def login(username, password):
    """
    Checks if entered credentials are correct.
    """

    if username == USERNAME and password == PASSWORD:
        st.session_state.logged_in = True
        return True

    return False


def logout():
    """
    Logs the user out.
    """

    st.session_state.logged_in = False


def check_login():
    """
    Creates login state if it does not exist.
    """

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    return st.session_state.logged_in