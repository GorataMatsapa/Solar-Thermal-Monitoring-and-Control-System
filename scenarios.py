"""
The six HIL tests as data, plus the pass/fail checks each one must satisfy.

Times are simulated seconds. Irradiance is a fixed, repeatable test profile (NOT weather).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import sev_rank


@dataclass(frozen=True)
class Injection:
    kind: str                      # "actuator_off" | "flow" | "sensor"
    start_s: float
    end_s: Optional[float]         # None = until the end of the run
    target: str = ""               # sensor id for kind == "sensor"
    mode: str = ""                 # sensor injection mode
    value: Optional[float] = None  # flow (L/min) or sensor value/offset
    label: str = ""

    def active(self, t):
        return t >= self.start_s and (self.end_s is None or t < self.end_s)


def profile(points):
    """Piecewise-linear profile from [(t_s, value), ...]."""
    xs, ys = zip(*points)
    return lambda t: float(np.interp(t, xs, ys))


@dataclass
class Scenario:
    sid: str
    name: str
    purpose: str
    duration_s: float
    irradiance: Callable
    injections: list = field(default_factory=list)
    mitigation_enabled: bool = True
    use_validated_inputs: bool = True
    ambient_c: float = 25.0
    tank_top_c: float = 50.0
    tank_bottom_c: float = 25.0
    collector_c: float = 25.0
    seed: int = 11
    checks: Callable = None        # fn(result) -> [(name, ok, detail)]


# ---------------------------------------------------------------------------
# checks (defined BEFORE the runs; thresholds come from the config, not from the results)
# ---------------------------------------------------------------------------

def _ev(res, ftype, subject=None):
    return [e for e in res.events if e["fault_type"] == ftype and (subject is None or e["subject"] == subject)]


def _checks_t1(res):
    df, cfg = res.df, res.cfg
    late = df[df.t_s >= res.scenario.duration_s / 3]
    return [
        ("no faults logged", len(res.events) == 0, f"{len(res.events)} events"),
        ("pump runs under sun", (late.pump_actual.mean() > 0.5), f"pump ON {late.pump_actual.mean():.0%} of the time after the first third of the test"),
        ("pump command == actual", bool((df.pump_cmd == df.pump_actual).all()), "command and actual never differ"),
        ("collector stays below warning level", df.twin_s1.max() < cfg.collector_c[0], f"max S1 {df.twin_s1.max():.1f} C"),
    ]


def _checks_t2(res):
    cfg, df = res.cfg, res.df
    ev = _ev(res, "PUMP_FAILURE")
    out = [("pump failure event logged", len(ev) == 1, f"{len(ev)} event(s)")]
    if not ev:
        return out
    e = ev[0]
    out += [
        ("source recorded as injected", e["source"] == "injected", e["source"]),
        ("detected within persistence + 2 steps", e["detection_delay_s"] <= cfg.pump_failure_persist_s + 2 * cfg.dt_s,
         f"{e['detection_delay_s']} s"),
        ("command ON / actual OFF / flow ~0 recorded",
         e["measured_at_detection"]["pump_command"] and not e["measured_at_detection"]["pump_actual"]
         and e["measured_at_detection"]["measured_flow_lpm"] < cfg.no_flow_lpm, str(e["measured_at_detection"])),
        ("SMS request logged (never falsely 'sent')", len(res.sms) >= 1 and all(s["status"] != "sent" for s in res.sms),
         ", ".join(sorted({s["status"] for s in res.sms})) or "none"),
    ]
    t0 = res.scenario.injections[0].start_s
    before = df[df.t_s < t0].twin_s1.iloc[-1]
    out.append(("collector temperature rises without circulation", df.twin_s1.iloc[-1] - before >= 30,
                f"S1 {before:.1f} -> {df.twin_s1.iloc[-1]:.1f} C"))
    return out


def _checks_t3(res):
    cfg, df = res.cfg, res.df
    ev = _ev(res, "LOW_FLOW")
    out = [("low-flow event logged", len(ev) == 1, f"{len(ev)} event(s)"),
           ("no pump-failure event (pump is really ON)", len(_ev(res, "PUMP_FAILURE")) == 0, "")]
    if not ev:
        return out
    e = ev[0]
    inj = res.scenario.injections[0]
    fault_rows = df[(df.t_s >= inj.start_s)]
    out += [
        ("detected only after the persistence time", cfg.low_flow_persist_s <= e["detection_delay_s"] <= cfg.low_flow_persist_s + 2 * cfg.dt_s,
         f"{e['detection_delay_s']} s (persistence {cfg.low_flow_persist_s:.0f} s)"),
        ("severity classified from deviation", e["peak_severity"] == "major",
         f"{e['measured_at_detection']['flow_deviation_pct']} % deviation -> {e['peak_severity']}"),
        ("pump command and actual stay ON", bool(fault_rows.pump_cmd.all() and fault_rows.pump_actual.all()), ""),
        ("deviation values recorded", e["measured_at_detection"]["flow_deviation_lpm"] is not None, str(e["measured_at_detection"])),
    ]
    return out


def _checks_t4(res):
    cfg = res.cfg
    ev = [e for e in res.events if e["fault_type"] in ("STAGNATION", "COLLECTOR_OVERTEMP")]
    out = [("stagnation event logged", len(ev) >= 1, f"{len(ev)} event(s)")]
    if not ev:
        return out
    e = ev[0]
    out.append(("severity reaches major or higher", sev_rank(e["peak_severity"]) >= 1, e["peak_severity"]))
    prot = [a for a in res.actions if a["domain"] == "SIMULATION"]
    if res.scenario.mitigation_enabled:
        out.append(("simulated protective actions applied", len(prot) >= 2, ", ".join(a["action"] for a in prot)))
        out.append(("protective actions are SIMULATION domain only",
                    all(a["domain"] != "PHYSICAL" for a in res.actions if a["action"].startswith("SIM_")), ""))
    else:
        out.append(("no protective actions in the unmitigated run", len(prot) == 0, f"{len(prot)} actions"))
    return out


def _checks_t5(res):
    cfg, df = res.cfg, res.df
    out = []
    for sid in ("S3", "S1", "S4"):
        ev = _ev(res, "SENSOR_FAULT", sid)
        inj = next(i for i in res.scenario.injections if i.target == sid)
        ok = len(ev) == 1 and ev[0]["detection_delay_s"] <= cfg.sensor_persist_s + 2 * cfg.dt_s
        out.append((f"{sid} fault detected and logged", ok,
                    f"{ev[0]['detection_delay_s']} s, {ev[0]['measured_at_detection']['reason']}" if ev else "missing"))
    if res.scenario.use_validated_inputs:
        out.append(("invalid readings never reached the controllers", res.metrics["invalid_inputs_used"] == 0,
                    f"{res.metrics['invalid_inputs_used']} samples"))
        out.append(("backup heater not switched on by the bad S3 reading", res.metrics["heater_on_s"] == 0,
                    f"heater ON {res.metrics['heater_on_s']:.0f} s"))
        out.append(("heater fail-safe OFF used when S3 is invalid beyond the hold time",
                    any(a["action"] == "HEATER_FAILSAFE_OFF" for a in res.actions), ""))
        out.append(("pump fallback used when S1/S4 invalid beyond the hold time",
                    any(a["action"] == "PUMP_FALLBACK_OFF" for a in res.actions), ""))
    else:
        out.append(("(raw run) bad readings DID reach the controllers", res.metrics["invalid_inputs_used"] > 0,
                    f"{res.metrics['invalid_inputs_used']} samples"))
    return out


def _checks_t6(res):
    cfg, df = res.cfg, res.df
    ev = _ev(res, "PUMP_FAILURE")
    out = [("pump failure event logged", len(ev) == 1, "")]
    if not ev:
        return out
    e = ev[0]
    inj = res.scenario.injections[0]
    out += [
        ("event marked RESOLVED, history retained", e["status"] == "RESOLVED" and e in res.events, e["status"]),
        ("recovery time recorded", e["recovery_s"] is not None and e["recovery_s"] >= inj.end_s,
         f"fault removed at {inj.end_s:.0f} s, resolved at {e['recovery_s']} s"),
        ("recovery within persistence + 2 steps", e["recovery_s"] - inj.end_s <= cfg.recovery_persist_s + 2 * cfg.dt_s, ""),
        ("pump command == actual at the end", bool(df.pump_cmd.iloc[-1] == df.pump_actual.iloc[-1] and df.pump_actual.iloc[-1]), ""),
        ("no active fault at the end", len([x for x in res.events if x["status"] != "RESOLVED"]) == 0, ""),
        ("collector cooling after recovery", df.twin_s1.iloc[-1] < df[df.t_s <= inj.end_s].twin_s1.iloc[-1],
         f"{df[df.t_s <= inj.end_s].twin_s1.iloc[-1]:.1f} -> {df.twin_s1.iloc[-1]:.1f} C"),
        ("recovery SMS request logged", any(s["kind"] == "recovery" for s in res.sms), ""),
    ]
    return out


def build_scenarios(kind="standard"):
    """
    kind "standard": the full-length tests (30-45 simulated minutes).
    kind "demo":     the same six tests squeezed into 15-20 simulated minutes, so that at 20x speed each
                        takes about a minute on the clock. Same faults, same detection rules and thresholds;
                        only the waiting times between steps are shorter and the sun is stronger.
    """
    if kind == "demo":
        return _demo_scenarios()
    sun800 = profile([(0, 0), (120, 800)])
    sun_boost = profile([(0, 0), (120, 800), (300, 800), (480, 1000)])
    mins = 60.0
    return {
        "T1": Scenario("T1", "Normal operation", "Baseline operation, no faults", 30 * mins, sun800,
                       checks=_checks_t1),
        "T2": Scenario("T2", "Pump failure / no circulation",
                       "Pump commanded ON, actuator forced OFF, flow zero", 40 * mins, sun800,
                       injections=[Injection("actuator_off", 10 * mins, None, label="pump actuator failure")],
                       checks=_checks_t2),
        "T3": Scenario("T3", "Low-flow condition",
                       "Pump ON but injected flow far below expected", 30 * mins, sun800,
                       injections=[Injection("flow", 10 * mins, None, value=0.6, label="injected low flow 0.6 L/min")],
                       checks=_checks_t3),
        "T4A": Scenario("T4A", "Stagnation - NO mitigation (Run A)",
                        "Pump disabled under high solar input, detection and logging only", 45 * mins, sun_boost,
                        injections=[Injection("actuator_off", 5 * mins, None, label="pump disabled (stagnation)")],
                        mitigation_enabled=False, checks=_checks_t4),
        "T4B": Scenario("T4B", "Stagnation - WITH simulated mitigation (Run B)",
                        "Same fault, autonomous simulated protection", 45 * mins, sun_boost,
                        injections=[Injection("actuator_off", 5 * mins, None, label="pump disabled (stagnation)")],
                        mitigation_enabled=True, checks=_checks_t4),
        "T5": Scenario("T5", "Sensor failure (validated inputs)",
                       "Disconnected / missing / implausible temperatures", 40 * mins, sun800,
                       injections=[
                           Injection("sensor", 8 * mins, 14 * mins, "S3", "disconnected", label="S3 disconnected (-127 C)"),
                           Injection("sensor", 18 * mins, 24 * mins, "S1", "dropout", label="S1 no data"),
                           Injection("sensor", 28 * mins, 34 * mins, "S4", "implausible", label="S4 implausible (250 C)")],
                       checks=_checks_t5),
        "T5R": Scenario("T5R", "Sensor failure - RAW inputs (comparison only)",
                        "Same faults but controllers fed unvalidated readings", 40 * mins, sun800,
                        injections=[
                            Injection("sensor", 8 * mins, 14 * mins, "S3", "disconnected", label="S3 disconnected (-127 C)"),
                            Injection("sensor", 18 * mins, 24 * mins, "S1", "dropout", label="S1 no data"),
                            Injection("sensor", 28 * mins, 34 * mins, "S4", "implausible", label="S4 implausible (250 C)")],
                        use_validated_inputs=False, checks=_checks_t5),
        "T6": Scenario("T6", "Fault recovery",
                       "Pump failure injected then removed; recovery monitored", 40 * mins, sun800,
                       injections=[Injection("actuator_off", 10 * mins, 20 * mins, label="pump actuator failure (removed)")],
                       checks=_checks_t6),
    }


def _demo_scenarios():
    mins = 60.0
    sun = profile([(0, 0), (60, 1000)])
    sun_boost = profile([(0, 800), (60, 1200)])       # strong afternoon sun; the system is already warm
    warm = dict(tank_top_c=70.0, tank_bottom_c=55.0, collector_c=85.0)
    sensor_inj = lambda t0, t1, tgt, mode, label: Injection("sensor", t0, t1, tgt, mode, label=label)
    return {
        "T1": Scenario("T1", "Normal operation", "Baseline operation, no faults", 15 * mins, sun, checks=_checks_t1),
        "T2": Scenario("T2", "Pump failure / no circulation",
                       "Pump commanded ON, actuator forced OFF, flow zero", 20 * mins, sun_boost,
                       injections=[Injection("actuator_off", 2 * mins, None, label="pump actuator failure")],
                       checks=_checks_t2, tank_top_c=60.0, tank_bottom_c=40.0, collector_c=50.0),
        "T3": Scenario("T3", "Low-flow condition",
                       "Pump ON but injected flow far below expected", 15 * mins, sun,
                       injections=[Injection("flow", 3 * mins, None, value=0.6, label="injected low flow 0.6 L/min")],
                       checks=_checks_t3),
        "T4A": Scenario("T4A", "Stagnation - NO mitigation (Run A)",
                        "Pump disabled under high solar input, detection and logging only", 20 * mins, sun_boost,
                        injections=[Injection("actuator_off", 60, None, label="pump disabled (stagnation)")],
                        mitigation_enabled=False, checks=_checks_t4, **warm),
        "T4B": Scenario("T4B", "Stagnation - WITH simulated mitigation (Run B)",
                        "Same fault, autonomous simulated protection", 20 * mins, sun_boost,
                        injections=[Injection("actuator_off", 60, None, label="pump disabled (stagnation)")],
                        mitigation_enabled=True, checks=_checks_t4, **warm),
        "T5": Scenario("T5", "Sensor failure (validated inputs)",
                       "Disconnected / missing / implausible temperatures", 20 * mins, sun,
                       injections=[sensor_inj(150, 330, "S3", "disconnected", "S3 disconnected (-127 C)"),
                                   sensor_inj(450, 630, "S1", "dropout", "S1 no data"),
                                   sensor_inj(750, 930, "S4", "implausible", "S4 implausible (250 C)")],
                       checks=_checks_t5),
        "T5R": Scenario("T5R", "Sensor failure - RAW inputs (comparison only)",
                        "Same faults but controllers fed unvalidated readings", 20 * mins, sun,
                        injections=[sensor_inj(150, 330, "S3", "disconnected", "S3 disconnected (-127 C)"),
                                    sensor_inj(450, 630, "S1", "dropout", "S1 no data"),
                                    sensor_inj(750, 930, "S4", "implausible", "S4 implausible (250 C)")],
                        use_validated_inputs=False, checks=_checks_t5),
        "T6": Scenario("T6", "Fault recovery",
                       "Pump failure injected then removed; recovery monitored", 20 * mins, sun,
                       injections=[Injection("actuator_off", 3 * mins, 8 * mins, label="pump actuator failure (removed)")],
                       checks=_checks_t6),
    }
