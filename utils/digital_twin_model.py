import numpy as np


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

WATER_DENSITY = 1.0          # kg/L
SPECIFIC_HEAT_WATER = 4186   # J/(kg·K)


class HOTSETDigitalTwin:
    """
    Simplified system-level digital twin of the
    HOTSET 300 L solar thermal system.

    Version 1:
    - Single-node tank temperature
    - Solar collector
    - Pump / flow
    - Hot-water demand
    - Tank heat loss
    - Backup heater
    - Environmental conditions

    The architecture is intentionally expandable for:
    - tank stratification
    - faults
    - HIL
    - predictive control
    - Firebase/live data
    """

    def __init__(
        self,
        volume_liters=300,
        initial_tank_temperature=25.0,
        ambient_temperature=20.0,
        cold_water_temperature=18.0,
        collector_temperature=25.0,
        solar_irradiance=500.0,
        flow_rate=0.0,
        hot_water_demand=0.0,
        heat_loss_coefficient=5.0,
        heater_power=0.0,
    ):

        # ----------------------------------------------------
        # SYSTEM PHYSICAL PROPERTIES
        # ----------------------------------------------------

        self.volume_liters = volume_liters

        self.mass = (
            volume_liters * WATER_DENSITY
        )

        # ----------------------------------------------------
        # TANK STATE
        # ----------------------------------------------------

        self.tank_temperature = initial_tank_temperature

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        self.ambient_temperature = ambient_temperature
        self.cold_water_temperature = cold_water_temperature

        # ----------------------------------------------------
        # SOLAR COLLECTOR
        # ----------------------------------------------------
        self.collector_temperature = collector_temperature

     # Configured peak irradiance for the simulation.
     # This remains constant while the simulated irradiance
     # changes with time.
        self.peak_solar_irradiance = solar_irradiance

     # Current irradiance at the present simulation time.
        self.solar_irradiance = solar_irradiance

        # Solar thermal power entering the tank.
        #
        # For Version 1 this can be supplied directly.
        # Later we will calculate it from collector area,
        # efficiency, irradiance and flow.
        self.solar_power = 0.0

        # ----------------------------------------------------
        # HYDRAULICS
        # ----------------------------------------------------

        self.flow_rate = flow_rate
        self.pump_on = False

        # ----------------------------------------------------
        # WATER DEMAND
        # ----------------------------------------------------

       # Hot-water demand in litres per second.
     #
      # This is separate from the solar-circuit flow rate,
     # which is displayed in L/min.
        self.hot_water_demand = hot_water_demand

        self.hot_water_temperature = (
            initial_tank_temperature
        )

        # ----------------------------------------------------
        # BACKUP HEATER
        # ----------------------------------------------------

        self.heater_power = heater_power
        self.heater_on = heater_power > 0

        # ----------------------------------------------------
        # THERMAL LOSSES
        # ----------------------------------------------------

        self.heat_loss_coefficient = (
            heat_loss_coefficient
        )

        # ----------------------------------------------------
        # DERIVED STATE
        # ----------------------------------------------------

        self.heat_loss = 0.0
        self.demand_power = 0.0
        self.stored_energy = 0.0

        self.update_derived_state()

    # ========================================================
    # SOLAR COLLECTOR
    # ========================================================

    def calculate_collector_temperature(self):
        """
        Simplified collector temperature model.

        Version 1:
        Collector temperature responds to solar irradiance
        and ambient temperature.

        This is intentionally simple and will later be
        replaced by a more physically meaningful collector
        model.
        """

        temperature_rise = (
            self.solar_irradiance * 0.04
        )

        return (
            self.ambient_temperature
            + temperature_rise
        )
    # ========================================================
    # SOLAR IRRADIANCE PROFILE
    # ========================================================

    def calculate_solar_irradiance(self, simulation_time_seconds):
        """
        Generate a simple variable solar irradiance profile.

        This is simulated environmental data for the digital twin.
        It is NOT real measured irradiance from the HOTSET system.

        The profile rises from zero, reaches a peak, and then
        decreases again.
        """

        time_minutes = simulation_time_seconds / 60.0

        # Normalised position through the simulation
        # 0.0 = beginning
        # 1.0 = end
        simulation_duration_minutes = 120.0

        position = (
            time_minutes
            / simulation_duration_minutes
        )

        # Keep the value within the simulation range
        position = np.clip(position, 0.0, 1.0)

        # Smooth bell-shaped solar profile
        solar_profile = np.sin(
            np.pi * position
        )

        return max(
            0.0,
            self.peak_solar_irradiance
            * solar_profile
        )
    # ========================================================
    # PUMP
    # ========================================================

    def update_pump_state(self):
        """
        Simple differential-temperature controller.

        The pump operates when the collector is sufficiently
        hotter than the tank.
        """

        temperature_difference = (
            self.collector_temperature
            - self.tank_temperature
        )

        if temperature_difference >= 5.0:
            self.pump_on = True
            self.flow_rate = 2.0

        elif temperature_difference <= 2.0:
            self.pump_on = False
            self.flow_rate = 0.0

    # ========================================================
    # SOLAR POWER
    # ========================================================

    def calculate_solar_power(self):
        """
        Simplified solar thermal power delivered to the tank.

        Q = m_dot * cp * ΔT

        For Version 1 we use the simulated collector/tank
        temperature difference and flow rate.
        """

        if not self.pump_on or self.flow_rate <= 0:
            return 0.0

      # flow_rate is expressed in L/min.
# Convert to kg/s for the thermal power equation:
#
# L/min → L/s → kg/s
        mass_flow_rate = (
          (self.flow_rate / 60.0)
          * WATER_DENSITY
        )

        temperature_difference = max(
            self.collector_temperature
            - self.tank_temperature,
            0.0
        )

        return (
            mass_flow_rate
            * SPECIFIC_HEAT_WATER
            * temperature_difference
        )

    # ========================================================
    # TANK HEAT LOSS
    # ========================================================

    def calculate_heat_loss(self):
        """
        Heat loss from the storage tank.
        """

        temperature_difference = (
            self.tank_temperature
            - self.ambient_temperature
        )

        return max(
            self.heat_loss_coefficient
            * temperature_difference,
            0.0
        )

    # ========================================================
    # HOT-WATER DEMAND
    # ========================================================

    def calculate_demand_power(self):
        """
        Energy removed from the tank by hot-water demand.

        hot_water_demand is litres/second.
        """

        if self.hot_water_demand <= 0:
            return 0.0

        temperature_difference = (
            self.tank_temperature
            - self.cold_water_temperature
        )

        if temperature_difference <= 0:
            return 0.0

        mass_flow_rate = (
            self.hot_water_demand
            * WATER_DENSITY
        )

        return (
            mass_flow_rate
            * SPECIFIC_HEAT_WATER
            * temperature_difference
        )

    # ========================================================
    # HOT-WATER OUTLET
    # ========================================================

    def calculate_hot_water_temperature(self):
        """
        Version 1 assumes outlet water is approximately
        equal to tank temperature.

        A mixer/tempering model will be added later once the
        real HOTSET hydraulic arrangement is verified.
        """

        return self.tank_temperature

    # ========================================================
    # STORED ENERGY
    # ========================================================

    def calculate_stored_energy(self):
        """
        Thermal energy stored in the tank relative to 0°C.
        """

        return (
            self.mass
            * SPECIFIC_HEAT_WATER
            * self.tank_temperature
        )

    def stored_energy_kwh(self):
        """
        Stored thermal energy in kWh.
        """

        return (
            self.calculate_stored_energy()
            / 3_600_000
        )

    # ========================================================
    # UPDATE DERIVED STATE
    # ========================================================

    def update_derived_state(self):

        self.heat_loss = (
            self.calculate_heat_loss()
        )

        self.demand_power = (
            self.calculate_demand_power()
        )

        self.hot_water_temperature = (
            self.calculate_hot_water_temperature()
        )

        self.stored_energy = (
            self.calculate_stored_energy()
        )

    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def step(
    self,
    timestep_seconds=60,
    simulation_time_seconds=0,
    ):
        """
        Advance the digital twin by one timestep.
        """

               # ----------------------------------------------------
        # 1. Update solar environment
        # ----------------------------------------------------

        self.solar_irradiance = (
            self.calculate_solar_irradiance(
                simulation_time_seconds
            )
        )

        # ----------------------------------------------------
        # 2. Update collector
        # ----------------------------------------------------

        self.collector_temperature = (
            self.calculate_collector_temperature()
        )

        # ----------------------------------------------------
        # 3. Update pump
        # ----------------------------------------------------

        self.update_pump_state()

        # ----------------------------------------------------
        # 4. Calculate solar heat input
        # ----------------------------------------------------

        self.solar_power = (
            self.calculate_solar_power()
        )

        # ----------------------------------------------------
        # 5. Calculate heat loss
        # ----------------------------------------------------

        self.heat_loss = (
            self.calculate_heat_loss()
        )

        # ----------------------------------------------------
        # 6. Calculate hot-water demand
        # ----------------------------------------------------

        self.demand_power = (
            self.calculate_demand_power()
        )

        # ----------------------------------------------------
        # 7. Calculate net thermal power
        # ----------------------------------------------------

        net_power = (
            self.solar_power
            + self.heater_power
            - self.heat_loss
            - self.demand_power
        )

        # ----------------------------------------------------
        # 8. Convert power into energy
        # ----------------------------------------------------

        energy_change = (
            net_power
            * timestep_seconds
        )

        # ----------------------------------------------------
        # 9. Convert energy into temperature change
        # ----------------------------------------------------

        temperature_change = (
            energy_change
            / (
                self.mass
                * SPECIFIC_HEAT_WATER
            )
        )

        self.tank_temperature += (
            temperature_change
        )

        # ----------------------------------------------------
        # 10. Prevent physically impossible temperatures
        # ----------------------------------------------------

        self.tank_temperature = max(
            self.tank_temperature,
            self.cold_water_temperature
        )

        # ----------------------------------------------------
        # 11. Update derived values
        # ----------------------------------------------------

        self.update_derived_state()

        return self.get_state()

    # ========================================================
    # SYSTEM STATE
    # ========================================================

    def get_state(self):
        """
        Return the complete digital-twin state.

        This dictionary will eventually be consumed by
        Streamlit, Firebase/HIL interfaces and control logic.
        """

        return {

            "tank": {
                "temperature": self.tank_temperature,
                "volume_liters": self.volume_liters,
                "stored_energy_kwh": self.stored_energy_kwh(),
            },

            "collector": {
                "temperature": self.collector_temperature,
                "solar_irradiance": self.solar_irradiance,
                "solar_power": self.solar_power,
            },

            "hydraulics": {
                "flow_rate": self.flow_rate,
                "pump_on": self.pump_on,
            },

            "water": {
                "cold_temperature": (
                    self.cold_water_temperature
                ),
                "hot_water_temperature": (
                    self.hot_water_temperature
                ),
                "demand": self.hot_water_demand,
            },

            "heater": {
                "on": self.heater_on,
                "power": self.heater_power,
            },

            "environment": {
                "ambient_temperature": (
                    self.ambient_temperature
                ),
            },

            "thermal": {
                "heat_loss": self.heat_loss,
                "demand_power": self.demand_power,
            },
        }


# ============================================================
# BATCH SIMULATION
# ============================================================

def run_simulation(
    duration_minutes=120,
    timestep_seconds=60,
    initial_temperature=25.0,
    ambient_temperature=20.0,
    cold_water_temperature=18.0,
    solar_irradiance=500.0,
    solar_power=None,
    hot_water_demand=0.0,
    heat_loss_coefficient=5.0,
    heater_power=0.0,
):
    """
    Run the HOTSET digital-twin simulation.

    Returns time-series data suitable for plotting.
    """
    # Backward compatibility with the previous model.
    # The old dashboard supplied solar_power directly.
    if solar_power is not None:
        solar_irradiance = solar_power

    model = HOTSETDigitalTwin(
        initial_tank_temperature=initial_temperature,
        ambient_temperature=ambient_temperature,
        cold_water_temperature=cold_water_temperature,
        solar_irradiance=solar_irradiance,
        hot_water_demand=hot_water_demand,
        heater_power=heater_power,
        heat_loss_coefficient=heat_loss_coefficient,
    )

    times = []
    tank_temperatures = []
    collector_temperatures = []
    stored_energy = []
    solar_power = []
    flow_rates = []
    pump_states = []

    cold_water_temperatures = []
    hot_water_temperatures = []
    heat_losses = []
    heater_states = []

    number_of_steps = int(
        duration_minutes
        * 60
        / timestep_seconds
    )

    for step in range(number_of_steps + 1):

        state = model.get_state()

        times.append(
            step
            * timestep_seconds
            / 60
        )

        tank_temperatures.append(
            state["tank"]["temperature"]
        )

        collector_temperatures.append(
            state["collector"]["temperature"]
        )

        stored_energy.append(
            state["tank"]["stored_energy_kwh"]
        )

        solar_power.append(
            state["collector"]["solar_power"]
        )

        flow_rates.append(
            state["hydraulics"]["flow_rate"]
        )

        pump_states.append(
            state["hydraulics"]["pump_on"]
        )

        cold_water_temperatures.append(
            state["water"]["cold_temperature"]
        )

        hot_water_temperatures.append(
            state["water"]["hot_water_temperature"]
        )

        heat_losses.append(
            state["thermal"]["heat_loss"]
        )

        heater_states.append(
            state["heater"]["on"]
        )

        model.step(
          timestep_seconds,
           simulation_time_seconds=(
          step * timestep_seconds
          ),
     )

       

    return {
    "time": np.array(times),

    # New system-level name
    "tank_temperature": np.array(
        tank_temperatures
    ),

    # Backward-compatible name for the
    # existing Digital Twin page
    "temperature": np.array(
        tank_temperatures
    ),

    "collector_temperature": np.array(
        collector_temperatures
    ),

    "stored_energy": np.array(
        stored_energy
    ),

    "solar_power": np.array(
        solar_power
    ),

    "flow_rate": np.array(
        flow_rates
    ),

     "pump_on": np.array(
        pump_states
    ),

    "cold_water_temperature": np.array(
        cold_water_temperatures
    ),

    "hot_water_temperature": np.array(
        hot_water_temperatures
    ),

    "heat_loss": np.array(
        heat_losses
    ),

    "heater_on": np.array(
        heater_states
    ),
}
    