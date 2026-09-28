import streamlit as st


def show():

    st.title("🤖 Predictive Control")

    st.info("This page is under development.")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()