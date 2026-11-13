"""
Autonomous mitigation.

Every action carries a DOMAIN so physical commands stay separate from simulation:

  PHYSICAL      a command that would go to real hardware (the pump relay -> fan)
  SIMULATION    acts on the digital twin only (simulated shading, simulated solenoid);
                NOT available on, and NOT validated for, the real HOTSET installation
  NOTIFICATION  alarm flag and SMS request
  SOFTWARE      validation / fallback behaviour inside the controller code
"""

from dataclasses import dataclass, field
from typing import Optional

from .config import sev_rank
from .detector import PUMP_FAILURE, LOW_FLOW, STAGNATION, COLLECTOR_OVERTEMP, SENSOR_FAULT, MANUAL_OVERRIDE

PHYSICAL, SIMULATION, NOTIFICATION, SOFTWARE = "PHYSICAL", "SIMULATION", "NOTIFICATION", "SOFTWARE"


@dataclass
class Action:
    t: float
    event_id: Optional[str]
    action: str
    domain: str
    detail: str

    def to_dict(self):
        return dict(self.__dict__)


class SMSNotifier:
    """
    Creates notification *requests* and (optionally) hands them to a sender.

    backend "unavailable": no channel integrated -> status "unavailable" (honest, never "sent")
    backend "firebase":    status "pending" until the ESP32/modem reports its result
    backend <sender>:      an internet sender from gateway.py (Twilio SMS or ntfy push); the status is whatever
                           the service reports: pending -> queued -> sent -> delivered, or failed
    """

    def __init__(self, backend="unavailable"):
        self.sender = None if isinstance(backend, str) else backend
        if self.sender is None and backend not in ("unavailable", "firebase"):
            raise ValueError("backend must be 'unavailable', 'firebase' or a sender object")
        self.backend = backend if self.sender is None else self.sender.name
        self.records = []

    def request(self, t, event, kind, text):
        waiting = "waiting for the modem to confirm" if self.sender is None else "sending ..."
        rec = {
            "sms_id": f"S-{event.event_id}-{kind}",
            "event_id": event.event_id, "kind": kind, "text": text[:160],
            "created_s": t, "updated_s": t,
            "channel": "sms" if self.sender is None else self.sender.channel,
            "status": "unavailable" if self.backend == "unavailable" else "pending",
            "detail": "no SMS channel connected" if self.backend == "unavailable" else waiting,
        }
        self.records.append(rec)
        event.sms.append({"sms_id": rec["sms_id"], "kind": kind, "status": rec["status"]})
        if self.sender is not None:
            import threading

            def update(status, detail):
                rec["status"], rec["detail"] = status, detail

            threading.Thread(target=self.sender.send, args=(rec["text"], update), daemon=True).start()
        return rec


def _fmt_time(t):
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def sms_text(event, kind):
    v = event.measured_at_detection
    head = f"STMaCS HIL TEST {event.fault_type} {event.severity.upper()}"
    if kind == "recovery":
        return f"{head.replace(event.severity.upper(), 'RESOLVED')} at t+{_fmt_time(event.recovery_s)}"
    if kind == "persisting":
        return f"{head} still active after protective action, please inspect at t+{_fmt_time(event.detection_s)}"
    bits = []
    if "collector_c" in v and v["collector_c"] is not None:
        bits.append(f"S1 {v['collector_c']}C")
    if "measured_flow_lpm" in v and v["measured_flow_lpm"] is not None:
        bits.append(f"flow {v['measured_flow_lpm']} L/min")
    if "sensor" in v:
        bits.append(f"{v['sensor']} {v['reason']}")
    return f"{head} at t+{_fmt_time(event.detection_s)} " + " ".join(bits)


class Mitigation:
    """Simulated shading + simulated solenoid are 'latched' until the cause is cleared."""

    def __init__(self, cfg, enabled, notifier):
        self.cfg, self.enabled, self.notifier = cfg, enabled, notifier
        self.actions = []
        self.shading_factor = 1.0          # SIMULATION-ONLY
        self.solenoid_open = False         # SIMULATION-ONLY
        self.alarm = False
        self.protection_latched = False
        self._cause_clear_since = None
        self._mitigated_at = {}
        self._persist_sms = set()

    def _log(self, t, event, action, domain, detail):
        a = Action(t, None if event is None else event.event_id, action, domain, detail)
        self.actions.append(a)
        if event is not None:
            event.mitigation_actions.append(a.to_dict())
        return a

    def _wants_sms(self, event, change):
        """Notify for faults with no autonomous fix, and for critical faults. Mitigated faults wait."""
        if event.severity == "critical":
            return not any(x["kind"] == "fault-critical" for x in event.sms)
        if change != "opened":
            return False
        mitigated_by_us = self.enabled and event.fault_type in (STAGNATION, COLLECTOR_OVERTEMP)
        return not mitigated_by_us

    def check_persistence(self, t, event):
        """A mitigated fault that is still open long after the protective action -> notify the crew once."""
        t0 = self._mitigated_at.get(event.event_id)
        if t0 is None or event.status == "RESOLVED" or event.event_id in self._persist_sms:
            return False
        if t - t0 < self.cfg.sms_persist_after_mitigation_s:
            return False
        self._persist_sms.add(event.event_id)
        rec = self.notifier.request(t, event, "persisting", sms_text(event, "persisting"))
        self._log(t, event, "SMS_REQUESTED", NOTIFICATION, f"fault persists after mitigation, status={rec['status']}")
        return True

    def on_change(self, t, event, change):
        cfg = self.cfg
        if event.fault_type == MANUAL_OVERRIDE:             # an operator action: logged, no alarm, no SMS
            if change == "opened":
                self._log(t, event, "OPERATOR_OVERRIDE", SOFTWARE,
                          f"operator forced the pump {event.measured_at_detection.get('manual_mode', '?').upper()}; "
                          "safety cut-outs still apply")
            elif change == "resolved":
                self._log(t, event, "OPERATOR_OVERRIDE_RELEASED", SOFTWARE, "back to automatic control")
            return
        if change in ("opened", "escalated"):
            if not self.alarm:
                self.alarm = True
            self._log(t, event, "ALARM_RAISED", NOTIFICATION, f"{event.fault_type} {event.severity}")
            if self._wants_sms(event, change):
                rec = self.notifier.request(t, event, f"fault-{event.severity}", sms_text(event, "fault"))
                self._log(t, event, "SMS_REQUESTED", NOTIFICATION, f"status={rec['status']}")
            if event.fault_type == SENSOR_FAULT and change == "opened":
                self._log(t, event, "VALIDATION_FALLBACK", SOFTWARE,
                          f"{event.subject} marked unreliable; invalid value blocked from controllers")
            if (self.enabled and event.fault_type in (STAGNATION, COLLECTOR_OVERTEMP)
                    and sev_rank(event.severity) >= sev_rank(cfg.protective_action_at)
                    and not self.protection_latched):
                self.protection_latched = True
                self.shading_factor = cfg.solar_reduction_factor
                self.solenoid_open = True
                event.status = "MITIGATING"
                self._mitigated_at[event.event_id] = t
                self._log(t, event, "SIM_SOLAR_INPUT_REDUCED", SIMULATION,
                          f"simulated collector shading, irradiance x{cfg.solar_reduction_factor}")
                self._log(t, event, "SIM_SOLENOID_OPEN", SIMULATION,
                          f"simulated heat-dump valve, {cfg.solenoid_dump_w_per_k} W/K (not a real HOTSET component)")
        elif change == "resolved":
            if event.sms:                      # only tell the crew "resolved" if they were told about the fault
                rec = self.notifier.request(t, event, "recovery", sms_text(event, "recovery"))
                self._log(t, event, "SMS_REQUESTED", NOTIFICATION, f"recovery, status={rec['status']}")

    def release_if_cleared(self, t, cause_cleared, stagnation_open):
        """Release latched simulated protection once the cause is gone for the recovery time."""
        if not self.protection_latched:
            return
        if cause_cleared and not stagnation_open:
            if self._cause_clear_since is None:
                self._cause_clear_since = t
            if t - self._cause_clear_since >= self.cfg.recovery_persist_s:
                self.protection_latched = False
                self.shading_factor = 1.0
                self.solenoid_open = False
                self._cause_clear_since = None
                self._log(t, None, "SIM_PROTECTION_RELEASED", SIMULATION,
                          "cause cleared: simulated shading removed, simulated solenoid closed")
        else:
            self._cause_clear_since = None

    def log_pump_fallback(self, t, entering, reason):
        self._log(t, None, "PUMP_FALLBACK_OFF" if entering else "PUMP_FALLBACK_RELEASED", PHYSICAL,
                  reason)

    def log_heater_failsafe(self, t, entering, reason):
        self._log(t, None, "HEATER_FAILSAFE_OFF" if entering else "HEATER_FAILSAFE_RELEASED", PHYSICAL, reason)
