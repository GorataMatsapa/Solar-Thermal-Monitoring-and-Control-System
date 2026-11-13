import math

import numpy as np


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

WATER_DENSITY = 1.0          # kg/L
SPECIFIC_HEAT_WATER = 4186   # J/(kg·K)

# Pump control modes
PUMP_AUTO = "AUTO"               # differential-temperature controller decides
PUMP_MANUAL_ON = "MANUAL_ON"     # operator forces the pump on
PUMP_MANUAL_OFF = "MANUAL_OFF"   # operator forces the pump off
PUMP_MODES = (PUMP_AUTO, PUMP_MANUAL_ON, PUMP_MANUAL_OFF)


class HOTSETDigitalTwin:
    """
    System-level digital twin of the HOTSET 300 L solar thermal system.

    Model structure
    ---------------
    - LAYERED storage tank: `tank_nodes` (default 6) equal-volume layers,
      index 0 = top ... index n-1 = bottom.
        * hot water is drawn from the TOP layer; cold mains water enters at the
          BOTTOM and pushes the layers upward (plug flow with some smearing),
        * the solar coil sits in the lower half of the tank, hot fluid entering
          at the top of the coil and leaving cooled at the bottom,
        * the backup heater element heats one layer (default: the middle one),
        * heat loss to ambient is shared equally between the layers,
        * a small conduction term and buoyancy mixing (a hotter layer below a
          colder one rises and mixes) keep the profile physical.
    - Lumped flat-plate collector with thermal mass, using the standard
      efficiency curve:   Q = A * (eta0 * G - a1 * dT - a2 * dT^2)
      so the collector responds to the pump (it cools down when fluid flows
      through it) and cannot deliver more energy than it absorbs.
    - Solar pump with differential-temperature control (hysteresis) that
      compares the collector with the BOTTOM tank sensor (S4), as on a real
      solar controller; manual override (AUTO / MANUAL_ON / MANUAL_OFF); and a
      tank high-limit cut-out on the TOP sensor (S3) that always wins.
    - Backup electric heater with thermostat (reads the middle sensor, S2)
      and hysteresis.
    - Hot-water demand, cold-water make-up.
    - Four temperature sensors (active in both modes; the live feed will supply
      the same signals on the real rig):
        S1  collector fluid on the hot line
        S2  tank, middle layer     S3  tank, top layer     S4  tank, bottom layer
      The displayed tank temperature is the average of S2, S3 and S4.
    - Two virtual heat meters (SIMULATION only, not part of live telemetry):
        HM1  solar loop   - flow sensor on the cooled return line, supply and
                            return temperature sensors either side of the coil
        HM2  domestic     - flow on the cold inlet, hot-out (top) vs cold-in

    Units
    -----
    temperatures: °C        irradiance: W/m²        power: W
    solar-circuit flow_rate: L/min                   hot_water_demand: L/s
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
        # --- collector (typical flat-plate values: adjust to the datasheet) ---
        collector_area=4.0,                    # m²
        collector_efficiency=0.75,             # eta0, optical efficiency
        collector_a1=4.0,                      # W/(m²·K)
        collector_a2=0.015,                    # W/(m²·K²)
        collector_heat_capacity_per_m2=6000.0,  # J/(m²·K)
        heat_exchanger_effectiveness=0.8,      # whole solar coil, hot in -> cooled out
        # --- pump / control ---
        pump_flow_rate=2.0,                    # L/min when running
        pump_dt_on=5.0,                        # K: switch ON above this difference
        pump_dt_off=2.0,                       # K: switch OFF below this difference
        tank_max_temperature=90.0,             # °C: high-limit cut-out (top sensor)
        tank_limit_reset_band=5.0,             # K below the limit before restart
        # --- backup heater ---
        heater_setpoint=55.0,                  # °C
        heater_hysteresis=3.0,                 # K
        # --- tank layers ---
        tank_nodes=6,                          # equal-volume layers, 0 = top
        solar_coil_nodes=None,                 # default: lower half of the tank
        heater_node=None,                      # default: middle layer (above the coil)
        node_conductance=2.0,                  # W/K between neighbouring layers
        buoyancy_swap=0.75,                    # 0.5 = rising water mixes fully, 1 = rises without mixing
        # --- simulation ---
        simulation_duration_minutes=120.0,
    ):

        # HIL / TEST HOOKS - every hook defaults to "off", so the existing pages
        # behave exactly as before. They are set by the HIL scenario engine.
        self.irradiance_override = None        # W/m2: replaces the bell curve when not None
        self.actuator_forced_off = False       # simulated pump actuator failure (command stays ON)
        self.flow_override = None              # L/min: replaces the pump flow while the pump runs
        self.solenoid_open = False             # SIMULATION-ONLY diversion / heat-dump valve
        self.solenoid_dump_w_per_k = 60.0      # EXPERIMENTAL: heat rejected per K of collector excess
        self.control_collector_override = None  # degC the pump controller sees instead of the true S1
        self.control_bottom_override = None     # degC the pump controller sees instead of the true S4

        # SYSTEM
        self.volume_liters = volume_liters
        self.mass = volume_liters * WATER_DENSITY

        # TANK (layered)
        self.tank_nodes = max(3, int(tank_nodes))
        n = self.tank_nodes
        self.node_volume = volume_liters / n
        self.node_heat_capacity = self.node_volume * WATER_DENSITY * SPECIFIC_HEAT_WATER
        self.node_temperatures = np.full(n, float(initial_tank_temperature))
        self.top_node = 0
        self.bottom_node = n - 1
        self.middle_node = (n - 1) // 2
        if solar_coil_nodes is None:
            solar_coil_nodes = range(n // 2, n)
        self.solar_coil_nodes = tuple(sorted(int(i) for i in solar_coil_nodes))
        if not self.solar_coil_nodes or not all(0 <= i < n for i in self.solar_coil_nodes):
            raise ValueError("solar_coil_nodes must be valid layer indices")
        self.heater_node = self.middle_node if heater_node is None else int(heater_node)
        if not 0 <= self.heater_node < n:
            raise ValueError("heater_node must be a valid layer index")
        self.node_conductance = max(0.0, float(node_conductance))
        self.buoyancy_swap = min(max(float(buoyancy_swap), 0.5), 1.0)
        self.solar_return_temperature = float(initial_tank_temperature)

        # ENVIRONMENT
        self.ambient_temperature = ambient_temperature
        self.cold_water_temperature = cold_water_temperature

        # COLLECTOR
        self.collector_temperature = collector_temperature
        self.collector_area = collector_area
        self.collector_efficiency = collector_efficiency
        self.collector_a1 = collector_a1
        self.collector_a2 = collector_a2
        self.collector_heat_capacity = collector_heat_capacity_per_m2 * collector_area
        self.heat_exchanger_effectiveness = heat_exchanger_effectiveness

        # Peak irradiance of the simulated day (can be changed live, e.g. to
        # imitate cloud cover) and the irradiance right now.
        self.peak_solar_irradiance = solar_irradiance
        self.simulation_duration_minutes = simulation_duration_minutes
        self.time_seconds = 0.0
        self.solar_irradiance = self.calculate_solar_irradiance(0.0)

        self.solar_power = 0.0          # W delivered to the tank (negative = losing heat)
        self.solar_energy_j = 0.0       # cumulative energy delivered to the tank

        # HEAT METERS (cumulative registers, like the totaliser on a real meter)
        self.hm_solar_energy_j = 0.0      # HM1: only counts positive (useful) heat
        self.hm_domestic_energy_j = 0.0   # HM2: heat drawn off with the hot water

        # HYDRAULICS / PUMP CONTROL
        self.flow_rate = flow_rate
        self.pump_on = flow_rate > 0           # ACTUAL pump state (what the fan/relay shows)
        self.pump_command = self.pump_on       # what the controller WANTS (differs only in a fault)
        self.pump_flow_rate = pump_flow_rate
        self.pump_dt_on = pump_dt_on
        self.pump_dt_off = pump_dt_off
        self.pump_mode = PUMP_AUTO
        self.tank_max_temperature = tank_max_temperature
        self.tank_limit_reset_band = tank_limit_reset_band
        self.high_limit_active = False

        # WATER DEMAND (litres per second, separate from solar-loop flow)
        self.hot_water_demand = hot_water_demand
        self.hot_water_temperature = initial_tank_temperature

        # BACKUP HEATER
        self.heater_power = heater_power
        self.heater_setpoint = heater_setpoint
        self.heater_hysteresis = heater_hysteresis
        self.heater_on = False

        # LOSSES
        self.heat_loss_coefficient = heat_loss_coefficient

        # DERIVED STATE
        self.heat_loss = 0.0
        self.demand_power = 0.0
        self.stored_energy = 0.0
        self.update_derived_state()

    # ========================================================
    # TANK TEMPERATURES
    # ========================================================

    @property
    def tank_temperature(self):
        """Volume-average temperature of the whole tank (layers are equal volume)."""
        return float(np.mean(self.node_temperatures))

    @property
    def top_temperature(self):
        return float(self.node_temperatures[self.top_node])

    @property
    def middle_temperature(self):
        return float(self.node_temperatures[self.middle_node])

    @property
    def bottom_temperature(self):
        return float(self.node_temperatures[self.bottom_node])

    # ========================================================
    # EXTERNAL INPUTS (safe to call at any time, e.g. from UI sliders)
    # ========================================================

    def set_ambient_temperature(self, value):
        self.ambient_temperature = float(value)

    def set_peak_irradiance(self, value):
        self.peak_solar_irradiance = max(0.0, float(value))

    def set_hot_water_demand(self, value):
        self.hot_water_demand = max(0.0, float(value))

    def set_heater(self, power, setpoint):
        self.heater_power = max(0.0, float(power))
        self.heater_setpoint = float(setpoint)
        self.update_heater_state()

    # --- HIL / test hooks (see __init__) ---
    def set_irradiance_override(self, value):
        """Hold irradiance at a fixed W/m2 (None = back to the bell curve)."""
        self.irradiance_override = None if value is None else max(0.0, float(value))
        self.solar_irradiance = self.calculate_solar_irradiance(self.time_seconds)

    def set_actuator_forced_off(self, forced):
        """Simulated pump failure: command unchanged, actual pump forced OFF."""
        self.actuator_forced_off = bool(forced)
        self.update_pump_state()

    def set_flow_override(self, litres_per_min):
        """Inject a circulation flow (L/min) while the pump runs; None = normal."""
        self.flow_override = None if litres_per_min is None else max(0.0, float(litres_per_min))
        self.update_pump_state()

    def set_solenoid(self, is_open):
        """SIMULATION-ONLY diversion valve. Not a model of any real HOTSET component."""
        self.solenoid_open = bool(is_open)

    def set_control_inputs(self, collector=None, bottom=None):
        """Temperatures the pump controller should use (validated sensor readings).
        None = use the true model temperature."""
        self.control_collector_override = None if collector is None else float(collector)
        self.control_bottom_override = None if bottom is None else float(bottom)

    def set_pump_mode(self, mode):
        """Switch between AUTO, MANUAL_ON and MANUAL_OFF and apply it immediately."""
        if mode not in PUMP_MODES:
            raise ValueError(f"pump mode must be one of {PUMP_MODES}")
        self.pump_mode = mode
        self.update_pump_state()
        if not self.pump_on:
            self.solar_power = 0.0

    # ========================================================
    # SOLAR IRRADIANCE PROFILE
    # ========================================================

    def calculate_solar_irradiance(self, simulation_time_seconds):
        """
        Simulated (NOT measured) bell-shaped irradiance: rises from zero,
        peaks mid-simulation, falls back to zero.
        """
        if self.irradiance_override is not None:
            return max(0.0, float(self.irradiance_override))
        position = (simulation_time_seconds / 60.0) / self.simulation_duration_minutes
        position = float(np.clip(position, 0.0, 1.0))
        return max(0.0, self.peak_solar_irradiance * math.sin(math.pi * position))

    # ========================================================
    # PUMP CONTROL
    # ========================================================

    def update_pump_state(self):
        """
        Priority (highest first):
          1. Tank high-limit cut-out   - top sensor (S3); pump forced OFF, even in manual
          2. Manual override           - MANUAL_ON / MANUAL_OFF
          3. Differential controller   - collector (S1) minus BOTTOM tank sensor (S4):
                                         ON at dT >= pump_dt_on, OFF at dT <= pump_dt_off
        """
        # 1. safety latch with hysteresis, on the hottest point of the tank
        if self.top_temperature >= self.tank_max_temperature:
            self.high_limit_active = True
        elif self.top_temperature <= (
            self.tank_max_temperature - self.tank_limit_reset_band
        ):
            self.high_limit_active = False

        command = self.pump_command
        if self.high_limit_active:
            command = False
        elif self.pump_mode == PUMP_MANUAL_ON:
            command = True
        elif self.pump_mode == PUMP_MANUAL_OFF:
            command = False
        else:
            collector = (self.collector_temperature if self.control_collector_override is None
                         else self.control_collector_override)
            bottom = (self.bottom_temperature if self.control_bottom_override is None
                      else self.control_bottom_override)
            difference = collector - bottom
            if difference >= self.pump_dt_on:
                command = True
            elif difference <= self.pump_dt_off:
                command = False
            # in between: keep the previous command (hysteresis)

        self.pump_command = command
        # HIL: an actuator fault keeps the COMMAND but forces the ACTUAL pump off
        self.pump_on = command and not self.actuator_forced_off

        flow = self.pump_flow_rate if self.pump_on else 0.0
        if self.pump_on and self.flow_override is not None:
            flow = max(0.0, float(self.flow_override))      # HIL: injected low flow
        self.flow_rate = flow

    # ========================================================
    # BACKUP HEATER (thermostat)
    # ========================================================

    def update_heater_state(self):
        """Thermostat reads the layer the element sits in (the middle sensor, S2, by default)."""
        if self.heater_power <= 0:
            self.heater_on = False
            return
        sensed = float(self.node_temperatures[self.heater_node])
        if sensed <= self.heater_setpoint - self.heater_hysteresis:
            self.heater_on = True
        elif sensed >= self.heater_setpoint:
            self.heater_on = False

    # ========================================================
    # SOLAR LOOP: COLLECTOR AND COIL
    # ========================================================

    def fluid_conductance(self):
        """Overall collector <-> coil conductance in W/K (m_dot*cp*effectiveness)."""
        if not self.pump_on or self.flow_rate <= 0:
            return 0.0
        mass_flow = (self.flow_rate / 60.0) * WATER_DENSITY  # L/min -> kg/s
        return (
            mass_flow
            * SPECIFIC_HEAT_WATER
            * self.heat_exchanger_effectiveness
        )

    def _coil_layer_effectiveness(self):
        """Per-layer effectiveness so the whole coil gives `heat_exchanger_effectiveness`."""
        eps = min(max(self.heat_exchanger_effectiveness, 0.0), 0.999)
        return 1.0 - (1.0 - eps) ** (1.0 / len(self.solar_coil_nodes))

    def coil_sink_temperature(self):
        """
        Effective temperature the solar fluid is cooled towards. The fluid runs
        down through the coil layers (hot end at the top), so the coolest, lowest
        layer weighs most. With a uniform tank this is simply the tank temperature.
        """
        e = self._coil_layer_effectiveness()
        eps_total = 1.0 - (1.0 - e) ** len(self.solar_coil_nodes)
        if eps_total <= 1e-9:
            return float(np.mean(self.node_temperatures[list(self.solar_coil_nodes)]))
        fluid = 0.0
        for idx in self.solar_coil_nodes:
            fluid = (1.0 - e) * fluid + e * self.node_temperatures[idx]
        return fluid / eps_total

    def update_collector(self, timestep_seconds):
        """
        Lumped collector node:

            C * dTc/dt = A*eta0*G - K_loss*(Tc - Ta) - K_fluid*(Tc - T_sink)

        solved with the exact exponential step so it stays stable for any
        timestep or flow rate. T_sink is the temperature the coil cools the
        fluid towards (see coil_sink_temperature).
        """
        ambient = self.ambient_temperature
        sink = self.coil_sink_temperature()
        delta = self.collector_temperature - ambient

        k_loss = self.collector_area * (
            self.collector_a1 + self.collector_a2 * abs(delta)
        )
        k_fluid = self.fluid_conductance()
        gain = (
            self.collector_area
            * self.collector_efficiency
            * self.solar_irradiance
        )

        # SIMULATION-ONLY: an open "solenoid" rejects collector heat to ambient
        k_dump = self.solenoid_dump_w_per_k if self.solenoid_open else 0.0

        k_total = k_loss + k_fluid + k_dump
        t_equilibrium = (gain + k_loss * ambient + k_fluid * sink + k_dump * ambient) / k_total
        tau = self.collector_heat_capacity / k_total

        self.collector_temperature = t_equilibrium + (
            self.collector_temperature - t_equilibrium
        ) * math.exp(-timestep_seconds / tau)

    def calculate_coil_heat(self):
        """
        Heat (W) the solar fluid gives to each tank layer. The fluid leaves the
        collector at its temperature and runs down through the coil layers, giving
        up heat to each in turn; the temperature it ends at is the loop return
        temperature (seen by the solar-loop heat meter). Negative if the pump
        circulates fluid colder than the tank (e.g. manual pump ON at night).
        """
        heat = np.zeros(self.tank_nodes)
        if self.fluid_conductance() <= 0:
            self.solar_return_temperature = self.bottom_temperature
            return heat

        m_cp = (self.flow_rate / 60.0) * WATER_DENSITY * SPECIFIC_HEAT_WATER
        e = self._coil_layer_effectiveness()
        fluid = self.collector_temperature
        for idx in self.solar_coil_nodes:
            q = m_cp * e * (fluid - self.node_temperatures[idx])
            heat[idx] = q
            fluid -= q / m_cp
        self.solar_return_temperature = fluid
        return heat

    def calculate_solar_power(self):
        """Total heat delivered to the tank by the loop (W)."""
        return float(self.calculate_coil_heat().sum())

    # ========================================================
    # TANK LAYER PHYSICS
    # ========================================================

    def _conduct(self, temperatures, timestep_seconds):
        """Slow heat flow between neighbouring layers (wall conduction, small mixing)."""
        if self.node_conductance <= 0 or self.tank_nodes < 2:
            return temperatures
        flux = self.node_conductance * (temperatures[:-1] - temperatures[1:])  # W, down
        change = np.zeros_like(temperatures)
        change[:-1] -= flux
        change[1:] += flux
        return temperatures + change * timestep_seconds / self.node_heat_capacity

    def _draw_off(self, temperatures, timestep_seconds):
        """
        Hot-water draw-off: water leaves from the top layer, cold mains water enters
        at the bottom and pushes every layer up by the volume drawn (first-order
        upwind scheme, sub-stepped so a layer is never over-emptied).
        Returns the new temperatures and the heat carried out of the tank (J).
        """
        volume = self.hot_water_demand * timestep_seconds   # litres
        if volume <= 0:
            return temperatures, 0.0

        fraction_total = volume / self.node_volume
        substeps = max(1, math.ceil(fraction_total / 0.5))
        fraction = fraction_total / substeps
        energy_out = 0.0
        for _ in range(substeps):
            energy_out += (
                fraction * self.node_volume * WATER_DENSITY * SPECIFIC_HEAT_WATER
                * (temperatures[0] - self.cold_water_temperature)
            )
            from_below = np.append(temperatures[1:], self.cold_water_temperature)
            temperatures = temperatures + fraction * (from_below - temperatures)
        return temperatures, energy_out

    def _remove_inversions(self, temperatures):
        """
        Buoyancy: a layer hotter than the one above it rises past it. `buoyancy_swap`
        sets how cleanly: 1.0 swaps the two layers (perfect stratification), 0.5 mixes
        them completely (uniform tank); in between a thermocline forms, as in a real
        cylinder. Energy is conserved (layers have equal volume).
        """
        a = self.buoyancy_swap
        t = temperatures.copy()
        for _ in range(100):
            changed = False
            for i in range(len(t) - 1):
                upper, lower = t[i], t[i + 1]
                if upper < lower - 1e-9:
                    t[i] = (1.0 - a) * upper + a * lower
                    t[i + 1] = a * upper + (1.0 - a) * lower
                    changed = True
            if not changed:
                break
        return t

    # ========================================================
    # TANK LOSSES, DEMAND, ENERGY
    # ========================================================

    def calculate_heat_loss(self):
        """Heat lost to ambient (W). Negative if ambient is warmer than the tank."""
        return self.heat_loss_coefficient * (
            self.tank_temperature - self.ambient_temperature
        )

    def calculate_demand_power(self):
        """Heat removed by hot-water draw-off right now (hot_water_demand in L/s)."""
        if self.hot_water_demand <= 0:
            return 0.0
        temperature_difference = (
            self.calculate_hot_water_temperature() - self.cold_water_temperature
        )
        if temperature_difference <= 0:
            return 0.0
        return (
            self.hot_water_demand
            * WATER_DENSITY
            * SPECIFIC_HEAT_WATER
            * temperature_difference
        )

    def calculate_hot_water_temperature(self):
        """Hot water leaves from the top of the tank (sensor S3)."""
        return self.top_temperature

    def calculate_stored_energy(self):
        """Thermal energy in the tank relative to 0 °C (J)."""
        return float(np.sum(self.node_heat_capacity * self.node_temperatures))

    def stored_energy_kwh(self):
        return self.calculate_stored_energy() / 3_600_000

    def useful_energy_kwh(self):
        """Energy above the cold-water temperature, i.e. what can actually be drawn off."""
        above_cold = np.maximum(self.node_temperatures - self.cold_water_temperature, 0.0)
        return float(np.sum(self.node_heat_capacity * above_cold)) / 3_600_000

    def collector_efficiency_now(self):
        """Instantaneous efficiency of the loop (delivered power / incident power)."""
        incident = self.collector_area * self.solar_irradiance
        if incident <= 1.0:
            return 0.0
        return self.solar_power / incident

    def update_derived_state(self):
        self.heat_loss = self.calculate_heat_loss()
        self.demand_power = self.calculate_demand_power()
        self.hot_water_temperature = self.calculate_hot_water_temperature()
        self.stored_energy = self.calculate_stored_energy()

    # ========================================================
    # TEMPERATURE SENSORS
    # ========================================================

    def tank_sensor_readings(self):
        """S1..S4 and the tank average (mean of S2, S3, S4) in °C."""
        middle = self.middle_temperature
        top = self.top_temperature
        bottom = self.bottom_temperature
        return {
            "s1_collector_fluid": self.collector_temperature,
            "s2_tank_middle": middle,
            "s3_tank_top": top,
            "s4_tank_bottom": bottom,
            "tank_average": (middle + top + bottom) / 3.0,
        }

    # ========================================================
    # HEAT METERS (virtual - simulation only)
    # ========================================================

    def heat_meter_readings(self):
        """
        What the two heat meters would display right now.

        HM1 (solar loop): supply temperature is the collector outlet, return
        temperature is the fluid leaving the bottom of the solar coil after it has
        given its heat to the tank layers. With the pump off there is no flow, so
        power is 0 (the stagnant return line is shown at the bottom-sensor
        temperature).
        Power = m_dot * cp * (T_supply - T_return), exactly the heat the model
        moves into the tank.

        HM2 (domestic): flow = hot-water draw-off, supply = hot out (top of the
        tank, S3), return = cold mains temperature.
        """
        # ---- HM1: solar loop ----
        solar_flow = self.flow_rate if self.pump_on else 0.0
        solar_supply = self.collector_temperature
        if solar_flow > 0:
            solar_return = self.solar_return_temperature
        else:
            solar_return = self.bottom_temperature
        solar_mass_flow = (solar_flow / 60.0) * WATER_DENSITY
        solar_power = (
            solar_mass_flow * SPECIFIC_HEAT_WATER * (solar_supply - solar_return)
        )

        # ---- HM2: domestic hot-water draw-off ----
        dhw_hot = self.hot_water_temperature
        dhw_cold = self.cold_water_temperature
        dhw_power = (
            self.hot_water_demand
            * WATER_DENSITY
            * SPECIFIC_HEAT_WATER
            * max(dhw_hot - dhw_cold, 0.0)
        )

        return {
            "solar_loop": {
                "flow_rate": solar_flow,                       # L/min
                "supply_temperature": solar_supply,            # °C
                "return_temperature": solar_return,            # °C
                "delta_t": solar_supply - solar_return,        # K
                "power": solar_power,                          # W
                "energy_kwh": self.hm_solar_energy_j / 3_600_000,
            },
            "domestic": {
                "flow_rate": self.hot_water_demand * 60.0,     # L/min
                "supply_temperature": dhw_hot,
                "return_temperature": dhw_cold,
                "delta_t": dhw_hot - dhw_cold,
                "power": dhw_power,
                "energy_kwh": self.hm_domestic_energy_j / 3_600_000,
            },
        }

    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def step(self, timestep_seconds=60, simulation_time_seconds=None):
        """
        Advance the twin by one timestep and return the new state.

        The model keeps its own clock, so callers normally pass only the
        timestep. `simulation_time_seconds` (time at the START of the step)
        is still accepted for backward compatibility.
        """
        if simulation_time_seconds is not None:
            self.time_seconds = simulation_time_seconds
        self.time_seconds += timestep_seconds
        dt = timestep_seconds

        # 1. environment at the end of this step
        self.solar_irradiance = self.calculate_solar_irradiance(self.time_seconds)

        # 2. controllers read last sample's temperatures (pump: collector vs S4)
        self.update_pump_state()
        self.update_heater_state()

        # 3. collector reacts to sun, ambient and flow
        self.update_collector(dt)

        # 4. heat flows into / out of each tank layer
        coil_heat = self.calculate_coil_heat()
        self.solar_power = float(coil_heat.sum())
        heater_watts = self.heater_power if self.heater_on else 0.0

        temperatures = self.node_temperatures.copy()
        loss_per_layer = (
            self.heat_loss_coefficient / self.tank_nodes
            * (temperatures - self.ambient_temperature)
        )
        self.heat_loss = float(loss_per_layer.sum())

        power = coil_heat - loss_per_layer
        power[self.heater_node] += heater_watts
        temperatures = temperatures + power * dt / self.node_heat_capacity

        # 5. layers exchange heat, hot water leaves / cold water enters, hot rises
        temperatures = self._conduct(temperatures, dt)
        temperatures, energy_out_j = self._draw_off(temperatures, dt)
        temperatures = self._remove_inversions(temperatures)

        # 6. physical sanity limits
        lowest = min(self.cold_water_temperature, self.ambient_temperature)
        self.node_temperatures = np.clip(temperatures, lowest, 100.0)

        self.demand_power = energy_out_j / dt
        self.solar_energy_j += self.solar_power * dt

        # 7. heat-meter totalisers (a heat meter only registers positive heat)
        self.hm_solar_energy_j += max(self.solar_power, 0.0) * dt
        self.hm_domestic_energy_j += max(energy_out_j, 0.0)

        self.update_derived_state()
        return self.get_state()

    # ========================================================
    # SYSTEM STATE
    # ========================================================

    def get_state(self):
        """Complete twin state for Streamlit, Firebase/HIL and control logic."""
        return {
            "tank": {
                "temperature": self.tank_temperature,
                "nodes": [float(t) for t in self.node_temperatures],
                "volume_liters": self.volume_liters,
                "stored_energy_kwh": self.stored_energy_kwh(),
                "useful_energy_kwh": self.useful_energy_kwh(),
            },
            "collector": {
                "temperature": self.collector_temperature,
                "solar_irradiance": self.solar_irradiance,
                "solar_power": self.solar_power,
                "efficiency": self.collector_efficiency_now(),
                "energy_delivered_kwh": self.solar_energy_j / 3_600_000,
            },
            "hydraulics": {
                "flow_rate": self.flow_rate,
                "pump_on": self.pump_on,
                "pump_command": self.pump_command,
                "actuator_forced_off": self.actuator_forced_off,
                "solenoid_open": self.solenoid_open,
                "pump_mode": self.pump_mode,
                "high_limit_active": self.high_limit_active,
            },
            "water": {
                "cold_temperature": self.cold_water_temperature,
                "hot_water_temperature": self.hot_water_temperature,
                "demand": self.hot_water_demand,
            },
            "heater": {
                "on": self.heater_on,
                "power": self.heater_power,
                "setpoint": self.heater_setpoint,
            },
            "environment": {
                "ambient_temperature": self.ambient_temperature,
            },
            "thermal": {
                "heat_loss": self.heat_loss,
                "demand_power": self.demand_power,
            },
            "heat_meters": self.heat_meter_readings(),
            "sensors": self.tank_sensor_readings(),
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
    **model_options,
):
    """
    Run the digital twin for `duration_minutes` and return time series for
    plotting. Extra keyword arguments (collector_area=..., pump_dt_on=...,
    tank_nodes=...) are passed straight to HOTSETDigitalTwin.

    The returned arrays have duration/timestep + 1 samples: index 0 is the
    initial state, index k is the state after k steps.
    """
    # Backward compatibility: the old dashboard supplied solar_power directly.
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
        simulation_duration_minutes=duration_minutes,
        **model_options,
    )

    series = {
        "time": [],
        "tank_temperature": [],
        "collector_temperature": [],
        "stored_energy": [],
        "solar_power": [],
        "flow_rate": [],
        "pump_on": [],
        "cold_water_temperature": [],
        "hot_water_temperature": [],
        "heat_loss": [],
        "heater_on": [],
        "solar_irradiance": [],
        # heat meters (useful as features when comparing with other installations)
        "hm_solar_flow": [],
        "hm_solar_supply_temperature": [],
        "hm_solar_return_temperature": [],
        "hm_solar_power": [],
        "hm_solar_energy_kwh": [],
        "hm_dhw_flow": [],
        "hm_dhw_power": [],
        "hm_dhw_energy_kwh": [],
        # temperature sensors
        "s1_collector_fluid": [],
        "s2_tank_middle": [],
        "s3_tank_top": [],
        "s4_tank_bottom": [],
        "tank_average": [],
    }

    def record(minutes, state):
        series["time"].append(minutes)
        series["tank_temperature"].append(state["tank"]["temperature"])
        series["collector_temperature"].append(state["collector"]["temperature"])
        series["stored_energy"].append(state["tank"]["stored_energy_kwh"])
        series["solar_power"].append(state["collector"]["solar_power"])
        series["flow_rate"].append(state["hydraulics"]["flow_rate"])
        series["pump_on"].append(state["hydraulics"]["pump_on"])
        series["cold_water_temperature"].append(state["water"]["cold_temperature"])
        series["hot_water_temperature"].append(state["water"]["hot_water_temperature"])
        series["heat_loss"].append(state["thermal"]["heat_loss"])
        series["heater_on"].append(state["heater"]["on"])
        series["solar_irradiance"].append(state["collector"]["solar_irradiance"])

        hm_solar = state["heat_meters"]["solar_loop"]
        hm_dhw = state["heat_meters"]["domestic"]
        series["hm_solar_flow"].append(hm_solar["flow_rate"])
        series["hm_solar_supply_temperature"].append(hm_solar["supply_temperature"])
        series["hm_solar_return_temperature"].append(hm_solar["return_temperature"])
        series["hm_solar_power"].append(hm_solar["power"])
        series["hm_solar_energy_kwh"].append(hm_solar["energy_kwh"])
        series["hm_dhw_flow"].append(hm_dhw["flow_rate"])
        series["hm_dhw_power"].append(hm_dhw["power"])
        series["hm_dhw_energy_kwh"].append(hm_dhw["energy_kwh"])

        for key, reading in state["sensors"].items():
            series[key].append(reading)

    number_of_steps = int(duration_minutes * 60 / timestep_seconds)

    record(0.0, model.get_state())
    for k in range(1, number_of_steps + 1):
        state = model.step(timestep_seconds)
        record(k * timestep_seconds / 60, state)

    result = {key: np.array(values) for key, values in series.items()}
    # Backward-compatible name used by the existing Digital Twin page.
    result["temperature"] = result["tank_temperature"]
    return result