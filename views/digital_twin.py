import streamlit as st
import pandas as pd

from utils.digital_twin_model import run_simulation


def show():

    st.title("🧩 HOTSET 300 L Digital Twin")

    st.markdown(
        """
        This digital twin represents a simplified thermal model of the
        300 L HOTSET solar thermal storage system.

        The model simulates the tank temperature based on solar heat input,
        ambient conditions, hot-water demand, thermal losses and optional
        backup heating.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Simulation parameters
    # ---------------------------------------------------------

    st.subheader("⚙️ Simulation Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_temperature = st.number_input(
            "Initial tank temperature (°C)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )

        ambient_temperature = st.number_input(
            "Ambient temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
        )

    with col2:
        solar_power = st.number_input(
            "Solar thermal input (W)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=100.0,
        )

        heat_loss = st.number_input(
            "Heat-loss coefficient (W/K)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )

    with col3:
        hot_water_demand = st.number_input(
            "Hot-water demand (L/s)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
        )

        heater_power = st.number_input(
            "Backup heater power (W)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=100.0,
        )

    duration = st.slider(
        "Simulation duration (minutes)",
        min_value=30,
        max_value=720,
        value=120,
        step=30,
    )

    st.divider()

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------

    if st.button(
        "▶ Run Digital Twin Simulation",
        type="primary",
        use_container_width=True,
    ):

        results = run_simulation(
            duration_minutes=duration,
            initial_temperature=initial_temperature,
            ambient_temperature=ambient_temperature,
            solar_power=solar_power,
            hot_water_demand=hot_water_demand,
            heat_loss_coefficient=heat_loss,
            heater_power=heater_power,
        )

        st.session_state.digital_twin_results = results

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    if "digital_twin_results" in st.session_state:

        results = st.session_state.digital_twin_results

        st.subheader("📊 Digital Twin State")

        # -----------------------------------------------------
        # Final simulated state
        # -----------------------------------------------------

        final_temperature = results["tank_temperature"][-1]
        final_energy = results["stored_energy"][-1]

        final_collector_temperature = (
            results["collector_temperature"][-1]
        )

        final_hot_water_temperature = (
            results["hot_water_temperature"][-1]
        )

        final_cold_water_temperature = (
            results["cold_water_temperature"][-1]
        )

        final_flow_rate = (
            results["flow_rate"][-1]
        )

        final_pump_state = (
            results["pump_on"][-1]
        )

        final_heater_state = (
            results["heater_on"][-1]
        )

        final_solar_power = (
            results["solar_power"][-1]
        )

        final_heat_loss = (
            results["heat_loss"][-1]
        )

        # -----------------------------------------------------
        # Row 1 — Thermal state
        # -----------------------------------------------------

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Tank Temperature",
            f"{final_temperature:.2f} °C",
        )

        metric2.metric(
            "Collector Temperature",
            f"{final_collector_temperature:.2f} °C",
        )

        metric3.metric(
            "Hot-Water Temperature",
            f"{final_hot_water_temperature:.2f} °C",
        )

        # -----------------------------------------------------
        # Row 2 — Hydraulic / control state
        # -----------------------------------------------------

        metric4, metric5, metric6 = st.columns(3)

        metric4.metric(
            "Flow Rate",
            f"{final_flow_rate:.2f} L/min",
        )

        metric5.metric(
            "Pump",
            "RUNNING" if final_pump_state else "OFF",
        )

        metric6.metric(
            "Backup Heater",
            "ON" if final_heater_state else "OFF",
        )

        # -----------------------------------------------------
        # Row 3 — Energy / water state
        # -----------------------------------------------------

        metric7, metric8, metric9 = st.columns(3)

        metric7.metric(
            "Stored Thermal Energy",
            f"{final_energy:.2f} kWh",
        )

        metric8.metric(
            "Solar Thermal Power",
            f"{final_solar_power:.1f} W",
        )

        metric9.metric(
            "Cold-Water Temperature",
            f"{final_cold_water_temperature:.2f} °C",
        )

        # -----------------------------------------------------
        # Additional thermal information
        # -----------------------------------------------------

        st.caption(
            f"Estimated tank heat loss: "
            f"{final_heat_loss:.1f} W"
        )

        st.divider()
        # -----------------------------------------------------
        # Tank temperature
        # -----------------------------------------------------

        st.subheader("🌡️ Tank Temperature")

        tank_data = pd.DataFrame(
            {
                "Time (minutes)": results["time"],
                "Tank Temperature (°C)": (
                    results["tank_temperature"]
                ),
            }
        )

        st.line_chart(
            tank_data.set_index("Time (minutes)")
        )

        # -----------------------------------------------------
        # Collector temperature
        # -----------------------------------------------------

        st.subheader("☀️ Collector Temperature")

        collector_data = pd.DataFrame(
            {
                "Time (minutes)": results["time"],
                "Collector Temperature (°C)": (
                    results["collector_temperature"]
                ),
            }
        )

        st.line_chart(
            collector_data.set_index("Time (minutes)")
        )

        # -----------------------------------------------------
        # Stored thermal energy
        # -----------------------------------------------------

        st.subheader("🔥 Stored Thermal Energy")

        energy_data = pd.DataFrame(
            {
                "Time (minutes)": results["time"],
                "Stored Energy (kWh)": (
                    results["stored_energy"]
                ),
            }
        )

        st.line_chart(
            energy_data.set_index("Time (minutes)")
        )

        # -----------------------------------------------------
        # Solar thermal power
        # -----------------------------------------------------

        st.subheader("☀️ Solar Thermal Power")

        solar_data = pd.DataFrame(
            {
                "Time (minutes)": results["time"],
                "Solar Thermal Power (W)": (
                    results["solar_power"]
                ),
            }
        )

        st.line_chart(
            solar_data.set_index("Time (minutes)")
        )

        # -----------------------------------------------------
        # Flow rate
        # -----------------------------------------------------

        st.subheader("💧 Solar Circuit Flow Rate")

        flow_data = pd.DataFrame(
            {
                "Time (minutes)": results["time"],
                "Flow Rate (L/min)": (
                    results["flow_rate"]
                ),
            }
        )

        st.line_chart(
            flow_data.set_index("Time (minutes)")
        )

    else:

        st.info(
            "Set the simulation parameters above and click "
            "'Run Digital Twin Simulation' to begin."
        )

    st.divider()

    # ---------------------------------------------------------
    # Back to home
    # ---------------------------------------------------------

    if st.button("⬅ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()