"""
HIL configuration.

EXPERIMENTAL HIL TEST SETTINGS
------------------------------
Every threshold below is a value chosen for the HIL demonstration. NONE of them is
a validated HOTSET safety limit. Change them here  and every module,
test and Firebase record follows.
"""

from dataclasses import dataclass, asdict

EXPERIMENTAL_LABEL = "EXPERIMENTAL HIL TEST SETTING - not a validated HOTSET safety limit"

SEVERITIES = ("warning", "major", "critical")      # ordered, lowest first


def sev_rank(severity):
    return SEVERITIES.index(severity)


@dataclass(frozen=True)
class HILConfig:
    # ---- simulation ----
    dt_s: float = 10.0                      # HIL step (s)
    noise_temp_c: float = 0.05              # seeded sensor noise (DS18B20 resolution is 0.0625 C)
    noise_flow_lpm: float = 0.03

    # ---- circulation ----
    expected_flow_lpm: float = 2.0          # = twin.pump_flow_rate
    no_flow_lpm: float = 0.2                # below this the loop counts as "no circulation"
    pump_failure_persist_s: float = 30.0    # command ON but actual OFF for this long -> fault
    low_flow_persist_s: float = 60.0
    low_flow_deviation_pct: tuple = (30.0, 60.0, 90.0)     # warning / major / critical

    # ---- collector stagnation / overtemperature ----
    collector_c: tuple = (100.0, 120.0, 140.0)             # warning / major / critical (S1)
    collector_hysteresis_k: float = 10.0    # an open event stays active until S1 < warning - this
    rate_window_s: float = 60.0
    rate_warn_k_per_min: float = 3.0        # early warning: fast rise with no circulation ...
    rate_check_min_c: float = 80.0          # ... once the collector is already above this
    stagnation_persist_s: float = 20.0

    # ---- sensor validation ----
    s1_range_c: tuple = (-20.0, 180.0)      # collector
    tank_range_c: tuple = (-5.0, 105.0)     # S2, S3, S4
    max_step_k_per_s: float = 1.0           # plausibility: allowed change rate ...
    jump_margin_k: float = 5.0              # ... plus this margin
    sensor_persist_s: float = 20.0          # unreliable for this long -> logged as a fault
    hold_last_good_s: float = 120.0         # a missing reading may be held this long, then it is "invalid"
    sensor_fail_pump_fallback: str = "OFF"  # S1/S4 invalid beyond the hold time (PHYSICAL relay command)

    # ---- operator manual override ----
    manual_override_persist_s: float = 10.0     # an override must last this long to be logged as an event

    # ---- who gets notified ----
    # SMS goes out for faults nobody fixes automatically (pump failure, low flow, sensor fault, stagnation
    # with no mitigation), for any fault that reaches "critical", and for a mitigated fault that is STILL
    # active this long after the protective action. A fault that mitigation clears does not send an SMS.
    sms_persist_after_mitigation_s: float = 300.0

    # ---- recovery / protection release ----
    recovery_persist_s: float = 60.0        # condition clear this long -> event resolved

    # ---- SIMULATION-ONLY protective actions ----
    protective_action_at: str = "major"     # severity at which simulated protection starts
    solar_reduction_factor: float = 0.3     # simulated shading/defocus of the collector
    solenoid_dump_w_per_k: float = 15.0     # simulated heat rejection through the "solenoid"

    # ---- existing hot-water controller (utils/stmacs_controller.py) ----
    heater_power_w: float = 5000.0


def config_as_dict(cfg: HILConfig) -> dict:
    d = asdict(cfg)
    d["_label"] = EXPERIMENTAL_LABEL
    return d
