"""
Sensor layer: fault injection on the readings, then validation.

  truth (twin)  ->  inject()  ->  measured  ->  SensorValidator  ->  validated
                         ^
        physical DS18B20 reading may replace the measured value of one sensor

Every reading keeps its source label so that physical and simulated/injected
values are never mixed silently.
"""

import math
from dataclasses import dataclass
from typing import Optional

SENSORS = ("S1", "S2", "S3", "S4")
DS18B20_DISCONNECTED_C = -127.0        # what a DS18B20 library returns for a missing sensor

# Injection modes for a temperature sensor
MODE_DISCONNECTED = "disconnected"     # reads -127 C like an unplugged DS18B20
MODE_DROPOUT = "dropout"               # no value at all (None / NaN)
MODE_IMPLAUSIBLE = "implausible"       # fixed absurd value
MODE_SPIKE = "spike"                   # truth + offset
MODE_OFFSET = "offset"                 # truth + offset (slow drift is not simulated)


def apply_sensor_injection(true_value: float, mode: str, value: Optional[float] = None):
    if mode == MODE_DISCONNECTED:
        return DS18B20_DISCONNECTED_C
    if mode == MODE_DROPOUT:
        return None
    if mode == MODE_IMPLAUSIBLE:
        return 250.0 if value is None else float(value)
    if mode in (MODE_SPIKE, MODE_OFFSET):
        return true_value + (40.0 if value is None else float(value))
    raise ValueError(f"unknown sensor injection mode {mode!r}")


@dataclass
class Reading:
    sensor: str
    raw: Optional[float]
    source: str                    # "simulated" | "injected" | "physical"
    reliable: bool
    reason: str                    # "" | "missing" | "out_of_range" | "jump"
    value: Optional[float]         # value the controllers may use (raw, held, or None)
    quality: str                   # "good" | "held" | "invalid"
    age_s: float = 0.0             # age of a held value


class SensorValidator:
    """Range, missing-value and rate-of-change checks, with a short hold of the last good value."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._last_good = {}       # sensor -> (t, value)

    def _range(self, sensor):
        return self.cfg.s1_range_c if sensor == "S1" else self.cfg.tank_range_c

    def check(self, t, sensor, raw, source) -> Reading:
        cfg = self.cfg
        reason = ""
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            reason = "missing"
        else:
            lo, hi = self._range(sensor)
            if not (lo <= raw <= hi):
                reason = "out_of_range"
            elif sensor in self._last_good:
                t0, v0 = self._last_good[sensor]
                if abs(raw - v0) > cfg.max_step_k_per_s * (t - t0) + cfg.jump_margin_k:
                    reason = "jump"

        if not reason:
            self._last_good[sensor] = (t, float(raw))
            return Reading(sensor, float(raw), source, True, "", float(raw), "good")

        last = self._last_good.get(sensor)
        if last is not None and (t - last[0]) <= cfg.hold_last_good_s:
            return Reading(sensor, raw, source, False, reason, last[1], "held", t - last[0])
        return Reading(sensor, raw, source, False, reason, None, "invalid",
                       (t - last[0]) if last else float("inf"))

    def update(self, t, measured: dict, sources: dict) -> dict:
        return {s: self.check(t, s, measured.get(s), sources.get(s, "simulated")) for s in SENSORS}


def raw_passthrough(measured: dict, sources: dict) -> dict:
    """Used only to demonstrate what happens WITHOUT validation (Test 5 comparison)."""
    out = {}
    for s in SENSORS:
        v = measured.get(s)
        out[s] = Reading(s, v, sources.get(s, "simulated"), True, "", v, "good")
    return out


def controller_inputs(readings: dict):
    """
    Values the hot-water controller may use.

    S3 good           -> used
    S3 held           -> last good value, for at most hold_last_good_s
    S3 invalid        -> None. The caller must then go to the FAIL-SAFE state
                         (backup heater OFF + alarm) instead of guessing a value.
    (S2 is NOT substituted for S3: S2 is cooler than S3, so it would make the heater
    run when it is not needed, and a guess is not a measurement.)
    """
    s2, s3 = readings["S2"].value, readings["S3"].value
    note = "S3 held at last good value" if (s3 is not None and readings["S3"].quality == "held") else ""
    return s3, s2, note
