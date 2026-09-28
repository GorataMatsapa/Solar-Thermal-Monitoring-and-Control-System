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


    # ------------------------------------------------------
    # LED CONTROL TEST
    # ------------------------------------------------------

    st.subheader("💡 LED Control Test")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Turn LED ON"):
            try:
                db_ref.child("stmacs").child("led_state").set(True)
                st.success("LED ON command sent to Firebase.")

            except Exception as e:
                st.error("Failed to send LED ON command.")
                st.exception(e)

    with col2:
        if st.button("🔴 Turn LED OFF"):
            try:
                db_ref.child("stmacs").child("led_state").set(False)
                st.success("LED OFF command sent to Firebase.")

            except Exception as e:
                st.error("Failed to send LED OFF command.")
                st.exception(e)