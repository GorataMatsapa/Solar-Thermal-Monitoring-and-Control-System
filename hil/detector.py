"""
Fault detection, quantification, severity and event bookkeeping.

evaluate()      pure function: readings -> which fault conditions are active right now,
                with severity and the measured/quantified values
EventManager    turns conditions into logged events: a condition must persist before an
                event opens, severity only escalates, an event is resolved (never deleted)
                once the condition has been clear for the recovery time.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .config import SEVERITIES, sev_rank

PUMP_FAILURE = "PUMP_FAILURE"
LOW_FLOW = "LOW_FLOW"
STAGNATION = "STAGNATION"
COLLECTOR_OVERTEMP = "COLLECTOR_OVERTEMP"
SENSOR_FAULT = "SENSOR_FAULT"
MANUAL_OVERRIDE = "MANUAL_OVERRIDE"       # the operator forces the pump ON or OFF from the dashboard


@dataclass
class Condition:
    active: bool
    severity: str = "warning"
    values: dict = field(default_factory=dict)


def _rnd(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def _severity_by_thresholds(x, thresholds):
    sev = "warning"
    for name, limit in zip(SEVERITIES, thresholds):
        if x >= limit:
            sev = name
    return sev


def evaluate(cfg, snap: dict, open_keys: set) -> dict:
    """
    snap keys: t, s1, s2, s3, s4 (validated values or None), readings (dict of Reading),
               pump_cmd, pump_actual, flow (measured L/min), rate_k_per_min
    Returns {(fault_type, subject): Condition}
    """
    out = {}
    cmd, actual, flow = snap["pump_cmd"], snap["pump_actual"], snap["flow"]
    expected = cfg.expected_flow_lpm if cmd else 0.0
    flow_known = flow is not None
    s1 = snap["s1"]

    def flow_values():
        dev = (expected - flow) if flow_known else None
        pct = (dev / expected * 100.0) if (flow_known and expected > 0) else None
        return {
            "expected_flow_lpm": _rnd(expected), "measured_flow_lpm": _rnd(flow),
            "flow_deviation_lpm": _rnd(dev), "flow_deviation_pct": _rnd(pct, 1),
            "pump_command": bool(cmd), "pump_actual": bool(actual),
        }

    # ---- 1. pump failure: commanded ON but the actuator is OFF ----
    if cmd and not actual:
        sev = "critical" if (s1 is not None and s1 >= cfg.collector_c[2]) else "major"
        out[(PUMP_FAILURE, "pump")] = Condition(True, sev, {**flow_values(), "collector_c": _rnd(s1)})
    else:
        out[(PUMP_FAILURE, "pump")] = Condition(False)

    # ---- 2. low flow: pump commanded ON and actually ON, but flow is far below expected ----
    low = False
    if cmd and actual and flow_known and expected > 0:
        pct = (expected - flow) / expected * 100.0
        if pct >= cfg.low_flow_deviation_pct[0]:
            low = True
            out[(LOW_FLOW, "flow")] = Condition(
                True, _severity_by_thresholds(pct, cfg.low_flow_deviation_pct), flow_values())
    if not low:
        out[(LOW_FLOW, "flow")] = Condition(False)

    # ---- 3. collector stagnation / overtemperature ----
    no_circulation = (flow < cfg.no_flow_lpm) if flow_known else (not actual)
    for ftype in (STAGNATION, COLLECTOR_OVERTEMP):
        out[(ftype, "S1")] = Condition(False)
    if s1 is not None:
        warn = cfg.collector_c[0]
        if (STAGNATION, "S1") in open_keys or (COLLECTOR_OVERTEMP, "S1") in open_keys:
            warn -= cfg.collector_hysteresis_k          # hysteresis: stay active while still hot
        rate = snap["rate_k_per_min"]
        hot = s1 >= warn
        rising_fast = (no_circulation and s1 >= cfg.rate_check_min_c
                       and rate is not None and rate >= cfg.rate_warn_k_per_min)
        if hot or rising_fast:
            ftype = STAGNATION if no_circulation else COLLECTOR_OVERTEMP
            sev = _severity_by_thresholds(s1, cfg.collector_c) if s1 >= cfg.collector_c[0] else "warning"
            tank_ref = snap["s3"] if snap["s3"] is not None else snap["s2"]
            out[(ftype, "S1")] = Condition(True, sev, {
                "collector_c": _rnd(s1), "tank_top_c": _rnd(snap["s3"]), "tank_bottom_c": _rnd(snap["s4"]),
                "collector_minus_tank_bottom_k": _rnd(s1 - snap["s4"]) if snap["s4"] is not None else None,
                "collector_minus_tank_top_k": _rnd(s1 - tank_ref) if tank_ref is not None else None,
                "rise_rate_k_per_min": _rnd(rate),
                "pump_command": bool(cmd), "pump_actual": bool(actual),
                "measured_flow_lpm": _rnd(flow),
                "trigger": "temperature" if hot else "rate_of_rise",
            })

    # ---- 3b. operator manual override (an operator ACTION, logged for traceability) ----
    mode = snap.get("manual_mode", "auto")
    if mode in ("on", "off"):
        s4 = snap["s4"]
        out[(MANUAL_OVERRIDE, "pump")] = Condition(True, "warning", {
            "manual_mode": mode, "pump_command": bool(cmd), "pump_actual": bool(actual),
            "collector_c": _rnd(s1), "tank_bottom_c": _rnd(s4),
            "collector_minus_tank_bottom_k": _rnd(s1 - s4) if (s1 is not None and s4 is not None) else None,
            "safety_fallback_active": bool(snap.get("pump_fallback")),
            "tank_high_limit_active": bool(snap.get("high_limit")),
            "note": "operator command; safety cut-outs (sensor fallback, tank high limit) still take priority",
        })
    else:
        out[(MANUAL_OVERRIDE, "pump")] = Condition(False)

    # ---- 4. sensor faults ----
    readings = snap["readings"]
    for sid, r in readings.items():
        if r.reliable:
            out[(SENSOR_FAULT, sid)] = Condition(False)
            continue
        if sid in ("S1", "S4"):
            sev = "major"                                    # pump control depends on them
        elif sid == "S3":
            sev = "warning" if readings["S2"].value is not None else "major"
        else:
            sev = "warning"
        out[(SENSOR_FAULT, sid)] = Condition(True, sev, {
            "sensor": sid, "reason": r.reason, "raw_value": _rnd(r.raw),
            "last_good_used": r.quality == "held", "quality": r.quality,
            "value_used_by_controller": _rnd(r.value), "source": r.source,
        })
    return out


# =============================================================================
# EVENTS
# =============================================================================

@dataclass
class FaultEvent:
    event_id: str
    fault_type: str
    subject: str
    source: str                           # "injected" | "naturally_detected"
    injections: list                      # labels of injections active at detection
    injection_start_s: Optional[float]    # earliest start of those injections
    first_condition_s: float              # first time the condition was seen (before persistence)
    detection_s: float                    # time the event was raised (after persistence)
    severity: str
    peak_severity: str
    severity_history: list
    measured_at_detection: dict
    measured_at_peak: dict
    mitigation_actions: list = field(default_factory=list)
    sms: list = field(default_factory=list)
    recovery_s: Optional[float] = None
    status: str = "ACTIVE"                # ACTIVE -> MITIGATING (if protection applied) -> RESOLVED
    duration_s: float = 0.0

    def to_dict(self, wall_start=None):
        d = dict(self.__dict__)
        if wall_start is not None:
            d["detection_epoch"] = wall_start + self.detection_s
            d["first_condition_epoch"] = wall_start + self.first_condition_s
            d["recovery_epoch"] = None if self.recovery_s is None else wall_start + self.recovery_s
        d["detection_delay_s"] = (None if self.injection_start_s is None
                                  else round(self.detection_s - self.injection_start_s, 1))
        d["time_to_recover_s"] = (None if self.recovery_s is None
                                  else round(self.recovery_s - self.detection_s, 1))
        return d


class EventManager:
    def __init__(self, cfg, run_id):
        self.cfg, self.run_id = cfg, run_id
        self.events = []                  # all events, resolved ones are kept
        self._open = {}                   # key -> FaultEvent
        self._pending = {}                # key -> first t the condition was seen
        self._clear_since = {}            # key -> first t the condition was clear
        self._n = 0

    def open_keys(self):
        return set(self._open)

    def open_events(self, *types):
        return [e for k, e in self._open.items() if not types or k[0] in types]

    def _persist(self, ftype):
        c = self.cfg
        return {PUMP_FAILURE: c.pump_failure_persist_s, LOW_FLOW: c.low_flow_persist_s,
                STAGNATION: c.stagnation_persist_s, COLLECTOR_OVERTEMP: c.stagnation_persist_s,
                SENSOR_FAULT: c.sensor_persist_s, MANUAL_OVERRIDE: c.manual_override_persist_s}[ftype]

    def update(self, t, conditions: dict, injections: list):
        """injections: [(label, start_s)] currently active. Returns [(event, change)]."""
        changes = []
        for key, cond in conditions.items():
            ev = self._open.get(key)
            if cond.active:
                self._clear_since.pop(key, None)
                if ev is None:
                    first = self._pending.setdefault(key, t)
                    if t - first >= self._persist(key[0]):
                        self._pending.pop(key)
                        self._n += 1
                        starts = [s for _, s in injections]
                        ev = FaultEvent(
                            event_id=f"F-{self.run_id}-{self._n:03d}", fault_type=key[0], subject=key[1],
                            source=("operator" if key[0] == MANUAL_OVERRIDE
                                    else "injected" if injections else "naturally_detected"),
                            injections=[lbl for lbl, _ in injections],
                            injection_start_s=min(starts) if starts else None,
                            first_condition_s=first, detection_s=t, severity=cond.severity,
                            peak_severity=cond.severity, severity_history=[(t, cond.severity)],
                            measured_at_detection=dict(cond.values), measured_at_peak=dict(cond.values))
                        self._open[key] = ev
                        self.events.append(ev)
                        changes.append((ev, "opened"))
                else:
                    ev.duration_s = t - ev.first_condition_s
                    if sev_rank(cond.severity) > sev_rank(ev.severity):
                        ev.severity = cond.severity
                        ev.severity_history.append((t, cond.severity))
                        if sev_rank(cond.severity) >= sev_rank(ev.peak_severity):
                            ev.peak_severity = cond.severity
                            ev.measured_at_peak = dict(cond.values)
                        changes.append((ev, "escalated"))
                    elif cond.severity == ev.peak_severity:
                        ev.measured_at_peak = dict(cond.values)
            else:
                self._pending.pop(key, None)
                if ev is not None:
                    since = self._clear_since.setdefault(key, t)
                    if t - since >= self.cfg.recovery_persist_s:
                        ev.recovery_s = t
                        ev.status = "RESOLVED"
                        ev.duration_s = t - ev.first_condition_s
                        del self._open[key]
                        self._clear_since.pop(key, None)
                        changes.append((ev, "resolved"))
        return changes
