"""
Live HIL bridge: Firebase  <->  digital twin + fault engine  <->  Firebase.

    ESP32  --writes-->  hil/physical     real DS18B20 / flow sensor readings, relay states
    Dashboard --writes-> hil/control     which test to run, and which sensors are REAL
    this program reads both, advances the SAME Simulation used by the test matrix, then writes
         hil/actuators   fan + valve commands for the ESP32 relays
         hil/live        everything the dashboard shows (labelled physical / simulated)
         hil/faults/...  fault events (history kept)      hil/sms/...   SMS requests

Nothing here deletes data. hil/live and hil/actuators are "latest value" nodes (overwritten on
purpose); faults, sms and runs are permanent records.

Time: one step = cfg.dt_s of simulated time. time_scale = how many simulated seconds pass per
real second (10 -> a 10-minute test takes about 1 minute).
"""

import time

from .config import HILConfig
from .runner import HILRunner
from .scenarios import build_scenarios
from .sinks import FirebaseSink, sanitize, NullSink

LIVE_SCENARIOS = ("T1", "T2", "T3", "T4A", "T4B", "T5", "T6", "T7")


class LiveBridge:
    def __init__(self, db, cfg: HILConfig = HILConfig(), time_scale=10.0, stale_after_s=15.0,
                 sms_backend="unavailable", clock=time.time, root="hil", write_records=True,
                 profile="standard"):
        self.db, self.cfg, self.root = db, cfg, root
        self.time_scale, self.stale_after_s = float(time_scale), float(stale_after_s)
        self.sms_backend, self.clock = sms_backend, clock
        self.sink = FirebaseSink(db, root) if write_records else NullSink()
        self.scenarios = build_scenarios(profile)
        self.profile = profile
        self.sim = None
        self.state = "idle"
        self.last_result = None
        self._last_cmd_id = None
        self._last_seq, self._last_seq_t = None, None
        self._seen_control = False
        self._net_down = False
        self._pump_id_floor = None             # operator pump commands older than the test start are ignored
        self._last_pump_id = None

    # ---------------- Firebase helpers ----------------
    def _get(self, path):
        return self.db.reference(f"{self.root}/{path}").get()

    def _set(self, path, value):
        self.db.reference(f"{self.root}/{path}").set(sanitize(value))

    # ---------------- inputs ----------------
    def read_hardware(self):
        """Returns (online, age_s, data). online = the ESP32's sequence number keeps changing."""
        data = self._get("physical") or {}
        now = self.clock()
        seq = data.get("seq")
        if seq is not None and seq != self._last_seq:
            self._last_seq, self._last_seq_t = seq, now
        age = None if self._last_seq_t is None else now - self._last_seq_t
        return (age is not None and age <= self.stale_after_s), age, data

    def read_control(self):
        c = self._get("control") or {}
        return {
            "cmd_id": c.get("cmd_id"), "scenario_cmd": c.get("scenario_cmd", "idle"),
            "s1_source": c.get("s1_source", "simulated"), "flow_source": c.get("flow_source", "simulated"),
            "pump_mode": c.get("pump_mode", "auto"), "pump_mode_id": c.get("pump_mode_id"),
        }

    # ---------------- scenario control ----------------
    def _start(self, sid):
        if sid not in LIVE_SCENARIOS:
            return False
        self._finish_current(aborted=self.sim is not None and not self.sim.done)
        runner = HILRunner(self.cfg, sink=self.sink, sms_backend=self.sms_backend, wall_start=self.clock())
        self.sim = runner.start(self.scenarios[sid])
        self.sim.manual_mode = "auto"
        self._pump_id_floor = self._last_pump_id
        self.state = "running"
        return True

    def _finish_current(self, aborted=False):
        if self.sim is not None and self.sim.rows:
            self.last_result = self.sim.finish()
        self.sim = None
        self.state = "stopped" if aborted else "finished"

    # ---------------- one loop iteration ----------------
    def tick(self, steps=1):
        """Read Firebase once, advance the simulation `steps` steps (catch-up when the network is slow),
        then publish once."""
        ctrl = self.read_control()
        # first read: remember the command id so a leftover command is not replayed at start-up
        if not self._seen_control:
            self._seen_control = True
            self._last_cmd_id = ctrl["cmd_id"]
        elif ctrl["cmd_id"] != self._last_cmd_id:
            self._last_cmd_id = ctrl["cmd_id"]
            if ctrl["scenario_cmd"] in LIVE_SCENARIOS:
                self._start(ctrl["scenario_cmd"])
            elif ctrl["scenario_cmd"] in ("stop", "idle"):
                self._finish_current(aborted=self.sim is not None and not self.sim.done)

        self._last_pump_id = ctrl["pump_mode_id"]
        if self.sim is not None and ctrl["pump_mode_id"] is not None and (
                self._pump_id_floor is None or ctrl["pump_mode_id"] > self._pump_id_floor):
            if ctrl["pump_mode"] in ("auto", "on", "off"):
                self.sim.manual_mode = ctrl["pump_mode"]      # operator button on the dashboard

        online, age, hw = self.read_hardware()
        use_phys_s1 = ctrl["s1_source"] == "physical" and online
        use_phys_flow = ctrl["flow_source"] == "physical" and online

        row = None
        if self.sim is not None:
            phys = None
            if use_phys_s1:
                phys = {"S1": hw.get("s1_c") if hw.get("s1_ok", True) else None}
            for _ in range(max(1, int(steps))):
                row = self.sim.step(phys=phys if phys is not None else {},
                                    phys_flow=hw.get("flow_lpm") if use_phys_flow else None)
                if self.sim.done:
                    self._finish_current()
                    break

        self._publish(ctrl, online, age, hw, use_phys_s1, use_phys_flow, row)
        return row

    # ---------------- outputs ----------------
    def _publish(self, ctrl, online, age, hw, use_phys_s1, use_phys_flow, row):
        sim = self.sim
        live_sim = sim if sim is not None else None
        fan = valve = False
        if live_sim is not None:
            fan = bool(live_sim.twin.pump_on)               # PHYSICAL relay -> fan  (actual pump state)
            valve = bool(live_sim.mit.solenoid_open)        # bench-demo valve; SIMULATION-ONLY action
        self._set("actuators", {
            "fan": fan, "valve": valve, "seq": int(self.clock() * 10) % 2_000_000_000,
            "updated_epoch": self.clock(), "source": "hil_bridge",
            "note": "fan = actual pump state; valve = simulated dump valve shown on a bench valve",
        })

        live = {
            "updated_epoch": self.clock(), "state": self.state, "time_scale": self.time_scale,
            "hardware": {
                "online": bool(online), "age_s": None if age is None else round(age, 1),
                "s1_requested": ctrl["s1_source"], "s1_effective": "physical" if use_phys_s1 else "simulated",
                "flow_requested": ctrl["flow_source"], "flow_effective": "physical" if use_phys_flow else "simulated",
                "physical_reading": {k: hw.get(k) for k in ("s1_c", "s1_ok", "flow_lpm", "fan_on", "valve_on", "rssi")},
            },
        }
        src = self.sim or None
        res = self.last_result
        if src is not None and row is not None:
            tw = src.twin
            live.update({
                "run_id": src.run_id, "scenario": src.sc.sid, "scenario_name": src.sc.name,
                "t_s": row["t_s"], "duration_s": src.sc.duration_s,
                "sensors": {
                    sid: {"value": None if row[f"val_{sid.lower()}"] != row[f"val_{sid.lower()}"] else round(float(row[f"val_{sid.lower()}"]), 2),
                          "source": row[f"src_{sid.lower()}"], "reliable": bool(row[f"rel_{sid.lower()}"])}
                    for sid in ("S1", "S2", "S3", "S4")},
                "twin": {"collector_c": round(row["twin_s1"], 2), "tank_top_c": round(row["twin_s3"], 2),
                         "tank_bottom_c": round(row["twin_s4"], 2), "irradiance_wm2": round(row["irradiance_wm2"], 1),
                         "shading_factor": row["shading_factor"]},
                "pump": {"command": bool(row["pump_cmd"]), "actual": bool(row["pump_actual"]),
                         "manual_mode": row["manual_mode"],
                         "fallback_off": bool(row["pump_fallback"])},
                "flow": {"measured_lpm": round(row["flow_meas_lpm"], 2), "source": row["flow_src"],
                         "expected_lpm": self.cfg.expected_flow_lpm},
                "heater": {"command": bool(row["heater_cmd"]), "on": bool(row["heater_on"])},
                "valve_SIM": {"open": bool(row["solenoid_open_SIM"])},
                "injections_active": row["injections_active"],
                "twin_view": self._twin_view(src, row),
                "faults": {"open": [self._brief(e) for e in src.events.open_events()],
                           "total": len(src.events.events)},
                "mitigation": {"protection_latched": bool(row["protection_latched"]),
                               "recent_actions": [a.to_dict() for a in src.mit.actions[-5:]]},
                "sms": {"requests": len(src.notifier.records),
                        "latest": src.notifier.records[-1] if src.notifier.records else None,
                        "recent": [{k: r.get(k) for k in ("sms_id", "event_id", "kind", "status", "detail", "channel", "text")}
                                   for r in src.notifier.records[-6:]]},
            })
        elif res is not None:
            live.update({"run_id": res.run_id, "scenario": res.scenario.sid, "scenario_name": res.scenario.name,
                         "last_run_metrics": res.metrics,
                         "faults": {"open": [], "total": len(res.events)}})
        self._set("live", live)
        self._set("status", {"state": self.state, "last_run_id": (res.run_id if res else "none"),
                             "bridge_epoch": self.clock(), "data_source": "HIL_SIMULATED"})

    @staticmethod
    def _twin_view(src, row):
        """Full digital-twin state (same keys the Digital Twin page draws). The temperatures
        are the TWIN's values (simulated HOTSET 300 L); sensors S1-S4 are what the controllers
        were allowed to use (S1 may be the real DS18B20)."""
        st_ = src.twin.get_state()

        def sv(k):
            v = row[f"val_{k}"]
            return None if v != v else round(float(v), 2)
        sensors = {"s1_collector_fluid": sv("s1"), "s2_tank_middle": sv("s2"),
                   "s3_tank_top": sv("s3"), "s4_tank_bottom": sv("s4"),
                   "tank_average": st_["sensors"]["tank_average"]}
        return {
            "collector_temperature": st_["collector"]["temperature"], "tank_temperature": st_["tank"]["temperature"],
            "cold_water_temperature": st_["water"]["cold_temperature"],
            "hot_water_temperature": st_["water"]["hot_water_temperature"],
            "flow_rate": st_["hydraulics"]["flow_rate"], "pump_state": st_["hydraulics"]["pump_on"],
            "backup_heater_state": st_["heater"]["on"], "solar_thermal_power": st_["collector"]["solar_power"],
            "stored_energy": st_["tank"]["stored_energy_kwh"], "solar_irradiance": st_["collector"]["solar_irradiance"],
            "pump_mode": st_["hydraulics"]["pump_mode"], "high_limit_active": st_["hydraulics"]["high_limit_active"],
            "ambient_temperature": st_["environment"]["ambient_temperature"],
            "hot_water_demand": st_["water"]["demand"], "heater_power": st_["heater"]["power"],
            "heater_setpoint": st_["heater"]["setpoint"], "heat_meters": st_["heat_meters"], "sensors": sensors,
            "solenoid_open": bool(st_["hydraulics"].get("solenoid_open")),
            "pump_command": bool(st_["hydraulics"].get("pump_command")),
        }

    @staticmethod
    def _brief(e):
        return {"event_id": e.event_id, "fault_type": e.fault_type, "subject": e.subject,
                "detail": e.measured_at_detection.get("manual_mode"),
                "severity": e.severity, "status": e.status, "source": e.source, "detection_s": e.detection_s}

    def sim_minutes_hint(self):
        return "15-20 simulated minute" if self.profile == "demo" else "30-45 simulated minute"

    def minutes_hint(self, interval):
        sim_s = 1200.0 if self.profile == "demo" else 2400.0
        return sim_s / self.time_scale

    def _announce(self, row):
        """Print a line only when something changes, so the terminal shows the bridge is alive."""
        online = self._last_seq_t is not None and (self.clock() - self._last_seq_t) <= self.stale_after_s
        sc = self.sim.sc.sid if self.sim is not None else "-"
        faults = len(self.sim.events.open_events()) if self.sim is not None else 0
        snap = (self.state, sc, online, faults)
        if snap != getattr(self, "_last_snap", None):
            self._last_snap = snap
            print(time.strftime("%H:%M:%S"), f"test={sc} state={self.state} ESP32={'online' if online else 'OFFLINE'} open_faults={faults}")

    # ---------------- main loop ----------------
    def run_forever(self):
        interval = self.cfg.dt_s / self.time_scale
        print(f"HIL bridge running ({self.profile} tests): {self.time_scale:g}x time "
              f"(a {self.sim_minutes_hint()} test takes about {self.minutes_hint(interval):.0f} s). Ctrl+C to stop.")
        last = time.time()
        try:
            while True:
                now = time.time()
                running = self.sim is not None
                due = int((now - last) / interval) if running else (1 if now - last >= 1.0 else 0)
                if due <= 0:
                    time.sleep(0.02)
                    continue
                if due > 60:                       # far behind (very slow network): drop the backlog
                    due, last = 60, now
                else:
                    last = last + due * interval if running else now
                try:
                    row = self.tick(steps=due)
                    self._announce(row)
                    if self._net_down:
                        print(time.strftime("%H:%M:%S"), "connection to Firebase is back")
                        self._net_down = False
                except KeyboardInterrupt:
                    raise
                except Exception as exc:          # network drop, SSL error, Firebase hiccup: keep going
                    if not self._net_down:
                        print(time.strftime("%H:%M:%S"), "cannot reach Firebase, will keep retrying:",
                              type(exc).__name__)
                        self._net_down = True
                    time.sleep(3)
                    last = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self._set("actuators", {"fan": False, "valve": False, "seq": 0, "updated_epoch": self.clock(),
                                        "source": "hil_bridge_stopped"})
                print("bridge stopped; relays commanded OFF")
            except Exception:
                print("bridge stopped; could not reach Firebase to send OFF (the ESP32 switches itself OFF "
                      "after 30 s without commands)")
