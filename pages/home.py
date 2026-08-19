import streamlit as st

from utils.firebase import db_ref


def show():

    st.title("☀ Smart Solar Thermal Monitoring and Control System")

    st.success("Home page loaded successfully.")

    st.write(
        """
        This is where the main dashboard interface will be built.
        """
    )

    # ------------------------------------------------------
    # FIREBASE CONNECTION TEST
    # ------------------------------------------------------

    try:

        data = db_ref.get()

        st.success("🟢 Firebase connection successful")

        st.write("Database status:")

        if data is None:
            st.info("Firebase is connected, but the database is currently empty.")

        else:
            st.json(data)

    except Exception as e:

        st.error("🔴 Firebase connection failed")

        st.exception(e)