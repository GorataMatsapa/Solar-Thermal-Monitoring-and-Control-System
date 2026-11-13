"""
HIL runner.

Simulation   one scenario that can be advanced ONE STEP at a time (used by the live bridge)
HILRunner    runs a whole scenario in one go (used by the test matrix)

Both use the same loop body, so the live demo and the repeatable tests cannot drift apart.

    scenario injections ---> twin hooks (irradiance, pump actuator, flow, solenoid)
    twin step            ---> TRUE sensor values
    sensor injection     ---> MEASURED values   (a physical sensor may replace one of them)
    SensorValidator      ---> VALIDATED values  (invalid readings never reach a controller)
    pump controller      <--- validated S1 / S4   (the twin's own differential-T logic)
    hot-water controller <--- validated S3 / S2   (the existing utils.stmacs_controller)
    FaultDetector / EventManager --> events --> Mitigation --> actions (physical vs simulated)
"""

import math
import time as _time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from utils.digital_twin_model import HOTSETDigitalTwin, PUMP_AUTO, PUMP_MANUAL_OFF, PUMP_MANUAL_ON
from utils.stmacs_controller import Controller, HeaterModel

from .config import HILConfig, SEVERITIES
from .detector import (EventManager, evaluate, STAGNATION, COLLECTOR_OVERTEMP)
from .mitigation import Mitigation, SMSNotifier
from .sensors import (SENSORS, SensorValidator, apply_sensor_injection, controller_inputs,
                      raw_passthrough)
from .sinks import NullSink

FORCE_ON_SETPOINT_C = 95.0       # twin thermostat setpoint that never cuts the element (as in ai_closed_loop)
HEATER_OFF_SETPOINT_C = 55.0
DATA_SOURCE = "HIL_SIMULATED"


@dataclass
class RunResult:
    scenario: object
    cfg: HILConfig
    run_id: str
    wall_start: float
    df: pd.DataFrame
    events: list
    actions: list
    sms: list
    metrics: dict


def _valid_number(x, lo, hi):
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and lo <= x <= hi


class Simulation:
    def __init__(self, runner, sc):
        self.runner, self.sc = runner, sc
        cfg = self.cfg = runner.cfg
        self.sink = runner.sink
        self.dt = cfg.dt_s
        self.run_id = f"{sc.sid}-{datetime.fromtimestamp(runner.wall_start):%Y%m%d-%H%M%S}"
        self.rng = np.random.default_rng(sc.seed)

        twin = self.twin = HOTSETDigitalTwin(initial_tank_temperature=45.0, ambient_temperature=sc.ambient_c,
                                             cold_water_temperature=18.0, heater_power=0.0)
        twin.node_temperatures = np.linspace(sc.tank_top_c, sc.tank_bottom_c, twin.tank_nodes)
        twin.collector_temperature = sc.collector_c
        twin.pump_command = twin.pump_on = False
        twin.flow_rate = 0.0
        twin.solenoid_dump_w_per_k = cfg.solenoid_dump_w_per_k
        twin.update_derived_state()

        self.validator = SensorValidator(cfg)
        self.events = EventManager(cfg, self.run_id)
        self.notifier = SMSNotifier(runner.sms_backend)
        self.mit = Mitigation(cfg, sc.mitigation_enabled, self.notifier)
        self.heater_ctrl = Controller(HeaterModel(power_kw=cfg.heater_power_w / 1000.0, volume_l=300.0))
        self.clock0 = datetime(2026, 11, 3, 10, 0, 0)

        self.n_steps = int(round(sc.duration_s / self.dt))
        self.k = 0
        self.rows = []
        self.s1_hist = deque()
        self.validated = None                  # last step's readings feed this step's controllers
        self.pump_fallback = False
        self.heater_failsafe = False
        self.heater_cmd = False
        self.invalid_inputs = self.no_input_ticks = 0
        self.heater_on_s = 0.0
        self.last_measured = {}
        self.last_sources = {}
        self.last_flow_meas = 0.0
        self.last_flow_src = "simulated"
        self.active_injections = []
        self._sms_pushed = set()
        self.manual_mode = "auto"              # operator override: "auto" | "on" | "off"

    @property
    def done(self):
        return self.k >= self.n_steps

    @property
    def t_s(self):
        return self.k * self.dt

    # ------------------------------------------------------------------
    def step(self, phys: Optional[dict] = None, phys_flow: Optional[float] = None) -> dict:
        """
        Advance one step.
        phys       {"S1": 41.2 | None, ...} readings from REAL sensors for this step (None = the real
                   sensor has no valid reading). Those sensors are labelled "physical" and are
                   never fault-injected: a real fault stays a real fault.
        phys_flow  a REAL flow reading (L/min) to use instead of the simulated one.
        """
        cfg, dt, sc, twin, mit = self.cfg, self.dt, self.sc, self.twin, self.mit
        k = self.k
        t_in, t = k * dt, (k + 1) * dt
        active = [i for i in sc.injections if i.active(t_in)]
        self.active_injections = active
        kinds = {i.kind for i in active}

        # ---- 1. scenario + mitigation act on the twin (simulation inputs) ----
        twin.set_irradiance_override(sc.irradiance(t_in) * mit.shading_factor)
        twin.set_actuator_forced_off("actuator_off" in kinds)
        flow_inj = next((i.value for i in active if i.kind == "flow"), None)
        twin.set_flow_override(flow_inj)
        twin.set_solenoid(mit.solenoid_open)          # SIMULATION-ONLY

        # ---- 2. controllers read the previous step's (validated) sensor values ----
        validated = self.validated
        ctrl_s1 = ctrl_s4 = ctrl_s3 = None
        ctrl_used = {"S1": False, "S3": False, "S4": False}
        if validated is not None:
            view = validated if sc.use_validated_inputs else raw_passthrough(
                {s: validated[s].raw for s in SENSORS}, {s: validated[s].source for s in SENSORS})
            s1_in, s4_in = view["S1"].value, view["S4"].value
            want_fallback = sc.use_validated_inputs and (s1_in is None or s4_in is None)
            if want_fallback != self.pump_fallback:
                self.pump_fallback = want_fallback
                mit.log_pump_fallback(
                    t_in, want_fallback,
                    "S1/S4 invalid beyond the hold time: pump commanded OFF (relay)" if want_fallback
                    else "S1 and S4 valid again: back to differential-temperature control")
            if self.pump_fallback:
                twin.set_pump_mode(PUMP_MANUAL_OFF)
            else:
                twin.set_pump_mode({"on": PUMP_MANUAL_ON, "off": PUMP_MANUAL_OFF}.get(self.manual_mode, PUMP_AUTO))
                nan = float("nan")
                twin.set_control_inputs(nan if s1_in is None else s1_in, nan if s4_in is None else s4_in)
                ctrl_s1, ctrl_s4 = s1_in, s4_in
                ctrl_used["S1"] = ctrl_used["S4"] = True

            s3_in, s2_in, _ = controller_inputs(view) if sc.use_validated_inputs else (
                view["S3"].value, view["S2"].value, "")
            failsafe = sc.use_validated_inputs and s3_in is None
            if failsafe != self.heater_failsafe:
                self.heater_failsafe = failsafe
                mit.log_heater_failsafe(
                    t_in, failsafe,
                    "S3 invalid beyond the hold time: backup heater commanded OFF (fail-safe)" if failsafe
                    else "S3 valid again: hot-water controller resumed")
            if failsafe:
                self.heater_cmd = False                      # no valid input: heater OFF
                self.no_input_ticks += 1
            else:
                now_dt = self.clock0 + timedelta(seconds=t_in)
                top = float("nan") if s3_in is None else s3_in
                mid = None if s2_in is None else s2_in
                self.heater_cmd = self.heater_ctrl.step(now_dt, top, None, buffer_centre=mid).heater_on
                ctrl_s3 = s3_in
                ctrl_used["S3"] = True
        twin.set_heater(cfg.heater_power_w if self.heater_cmd else 0.0,
                        FORCE_ON_SETPOINT_C if self.heater_cmd else HEATER_OFF_SETPOINT_C)

        for sid, lo, hi, val in (("S1", *cfg.s1_range_c, ctrl_s1), ("S4", *cfg.tank_range_c, ctrl_s4),
                                 ("S3", *cfg.tank_range_c, ctrl_s3)):
            if ctrl_used[sid] and not _valid_number(val, lo, hi):
                self.invalid_inputs += 1
                break

        # ---- 3. physics ----
        # Bench mode: when the REAL DS18B20 stands in for S1, the twin's collector fluid IS that probe,
        # so warming the probe really heats the simulated collector (and the pump logic reacts to it).
        if phys is None:
            phys = self.runner.physical_inputs(t) if self.runner.physical_inputs else {}
        real_s1 = phys.get("S1")
        if real_s1 is not None and _valid_number(real_s1, *cfg.s1_range_c):
            twin.collector_temperature = float(real_s1)
        twin.step(dt)
        if twin.heater_on and twin.heater_power > 0:
            self.heater_on_s += dt

        # ---- 4. sensors: truth -> measured (injection / physical) ----
        truth = twin.tank_sensor_readings()
        true_v = {"S1": truth["s1_collector_fluid"], "S2": truth["s2_tank_middle"],
                  "S3": truth["s3_tank_top"], "S4": truth["s4_tank_bottom"]}
        measured, sources = {}, {}
        for sid in SENSORS:
            if sid in phys:
                measured[sid], sources[sid] = phys[sid], "physical"
                continue
            v = true_v[sid] + self.rng.normal(0.0, cfg.noise_temp_c)
            inj = next((i for i in active if i.kind == "sensor" and i.target == sid), None)
            if inj is not None:
                v = apply_sensor_injection(v, inj.mode, inj.value)
                sources[sid] = "injected"
            else:
                sources[sid] = "simulated"
            measured[sid] = v
        flow_true = twin.flow_rate
        if phys_flow is not None:
            flow_meas, flow_src = max(0.0, float(phys_flow)), "physical"
        else:
            flow_meas = max(0.0, flow_true + self.rng.normal(0.0, cfg.noise_flow_lpm)) if flow_true > 0 else 0.0
            flow_src = "injected" if flow_inj is not None else "simulated"
        self.last_measured, self.last_sources = measured, sources
        self.last_flow_meas, self.last_flow_src = flow_meas, flow_src

        validated = self.validated = self.validator.update(t, measured, sources)

        # ---- 5. detection ----
        s1v = validated["S1"].value if validated["S1"].reliable else None
        if s1v is not None:
            self.s1_hist.append((t, s1v))
        while self.s1_hist and t - self.s1_hist[0][0] > cfg.rate_window_s:
            self.s1_hist.popleft()
        rate = None
        if s1v is not None and len(self.s1_hist) >= 2 and (t - self.s1_hist[0][0]) >= 30.0:
            rate = (s1v - self.s1_hist[0][1]) / ((t - self.s1_hist[0][0]) / 60.0)

        snap = {"t": t, "s1": s1v, "s2": validated["S2"].value, "s3": validated["S3"].value,
                "s4": validated["S4"].value, "readings": validated, "pump_cmd": twin.pump_command,
                "pump_actual": twin.pump_on, "flow": flow_meas, "rate_k_per_min": rate,
                "manual_mode": self.manual_mode, "pump_fallback": self.pump_fallback,
                "high_limit": bool(getattr(twin, "high_limit_active", False))}
        conds = evaluate(cfg, snap, self.events.open_keys())
        changes = self.events.update(t, conds, [(i.label, i.start_s) for i in active])

        # ---- 6. mitigation ----
        for ev, change in changes:
            mit.on_change(t, ev, change)
            self.sink.upsert_event(ev.to_dict(self.runner.wall_start))
        for ev in self.events.open_events():
            if mit.check_persistence(t, ev):
                self.sink.upsert_event(ev.to_dict(self.runner.wall_start))
        flow_ok = flow_meas >= 0.7 * cfg.expected_flow_lpm
        cause_cleared = (twin.pump_command == twin.pump_on) and ((not twin.pump_command) or flow_ok)
        mit.release_if_cleared(t, cause_cleared, bool(self.events.open_events(STAGNATION, COLLECTOR_OVERTEMP)))

        for rec in self.notifier.records:                      # push new / changed notification statuses
            key = (rec["sms_id"], rec["status"])
            if key not in self._sms_pushed:
                self._sms_pushed.add(key)
                self.sink.upsert_sms(rec)

        open_sev = [SEVERITIES.index(e.severity) + 1 for e in self.events.open_events()]
        row = {"t_s": t, "irradiance_wm2": twin.solar_irradiance, "shading_factor": mit.shading_factor,
               "ambient_c": sc.ambient_c}
        for sid in SENSORS:
            low = sid.lower()
            row[f"twin_{low}"] = true_v[sid]
            row[f"meas_{low}"] = measured[sid] if measured[sid] is not None else np.nan
            row[f"val_{low}"] = validated[sid].value if validated[sid].value is not None else np.nan
            row[f"rel_{low}"] = int(validated[sid].reliable)
            row[f"src_{low}"] = sources[sid]
        row.update({
            "pump_cmd": int(twin.pump_command), "pump_actual": int(twin.pump_on),
            "flow_twin_lpm": flow_true, "flow_meas_lpm": flow_meas, "flow_src": flow_src,
            "heater_cmd": int(self.heater_cmd), "heater_on": int(twin.heater_on and twin.heater_power > 0),
            "solenoid_open_SIM": int(mit.solenoid_open), "protection_latched": int(mit.protection_latched),
            "manual_mode": self.manual_mode,
            "pump_fallback": int(self.pump_fallback), "rise_rate_k_min": np.nan if rate is None else rate,
            "open_events": len(open_sev), "max_open_severity": max(open_sev) if open_sev else 0,
            "injections_active": "|".join(i.label for i in active),
            "data_source": DATA_SOURCE,
        })
        self.rows.append(row)
        self.k += 1
        return row

    # ------------------------------------------------------------------
    def finish(self) -> RunResult:
        cfg, dt = self.cfg, self.dt
        df = pd.DataFrame(self.rows)
        ev_dicts = [e.to_dict(self.runner.wall_start) for e in self.events.events]
        for ed in ev_dicts:
            self.sink.upsert_event(ed)
        for rec in self.notifier.records:
            self.sink.upsert_sms(rec)

        above = df.twin_s1 >= cfg.collector_c[0]
        metrics = {
            "peak_collector_c": round(float(df.twin_s1.max()), 1),
            "t_peak_s": float(df.t_s[df.twin_s1.idxmax()]),
            "final_collector_c": round(float(df.twin_s1.iloc[-1]), 1),
            "seconds_collector_above_warning": float(above.sum() * dt),
            "heater_on_s": self.heater_on_s,
            "invalid_inputs_used": self.invalid_inputs,
            "controller_no_input_ticks": self.no_input_ticks,
            "events": len(ev_dicts),
            "events_resolved": sum(e["status"] == "RESOLVED" for e in ev_dicts),
        }
        result = RunResult(self.sc, cfg, self.run_id, self.runner.wall_start, df, ev_dicts,
                           [a.to_dict() for a in self.mit.actions], self.notifier.records, metrics)
        self.sink.write_run(result)
        return result


class HILRunner:
    def __init__(self, cfg: HILConfig = HILConfig(), sink=None, sms_backend="unavailable",
                 physical_inputs: Optional[Callable] = None, wall_start: Optional[float] = None):
        """
        physical_inputs(t_s) -> {"S1": 41.2, ...}: readings from REAL sensors, if any (batch use).
        The live bridge passes physical readings to Simulation.step() directly instead.
        """
        self.cfg = cfg
        self.sink = sink or NullSink()
        self.sms_backend = sms_backend
        self.physical_inputs = physical_inputs
        self.wall_start = wall_start if wall_start is not None else _time.time()

    def start(self, sc) -> Simulation:
        return Simulation(self, sc)

    def run(self, sc) -> RunResult:
        sim = Simulation(self, sc)
        while not sim.done:
            # a scripted operator (for the repeatable test matrix); the live demo uses real button presses
            sim.manual_mode = sc.operator_mode_at(sim.t_s)
            sim.step()
        return sim.finish()
