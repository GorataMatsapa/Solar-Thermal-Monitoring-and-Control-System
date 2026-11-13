"""
Where HIL records go.

NullSink       keeps nothing (the runner always returns everything in memory)
FirebaseSink   writes to the Realtime Database under hil/.  NOTHING IS EVER DELETED
               OR EXPIRED BY THIS CODE: no TTL, no cleanup, no overwrite of other
               paths. You remove records yourself, in the Firebase console.

Layout (all under  hil/ ):
  _schema                    short description of this layout (written once)
  config/thresholds          the experimental thresholds used (written once, then only by you)
  control                    dashboard -> HIL commands (defaults written once, never overwritten)
  status                     latest HIL run status
  physical/                  REAL hardware readings (the ESP32 writes here)       [later stage]
  injected/                  simulated / injected readings (Python writes here)   [later stage]
  twin/                      latest digital-twin state                            [later stage]
  faults/<event_id>          one record per fault event; resolved events stay
  sms/<sms_id>               SMS requests and their status
  runs/<run_id>/meta         one record per scenario run
  runs/<run_id>/series/<n>   the logged time series (every `series_stride`-th row)

Firebase does not store empty nodes, so faults/, sms/ and runs/ appear with the first
write. Physical and simulated readings live in different branches and every row also
carries data_source, so they cannot be confused with live_telemetry or System 455 data.
"""

import json
import math
import re

SCHEMA_VERSION = 1
_BAD_KEY = re.compile(r"[.$#\[\]/]")


def sanitize(obj):
    """Firebase-safe copy: NaN/inf -> removed, numpy -> python, tuples -> lists."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k = _BAD_KEY.sub("_", str(k))
            v = sanitize(v)
            if v is not None:
                out[k] = v
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class NullSink:
    def upsert_event(self, ev): pass
    def upsert_sms(self, rec): pass
    def write_run(self, result): pass


class FirebaseSink:
    def __init__(self, db, root="hil", series_stride=3):
        """db = firebase_admin.db (the module).  series_stride=3 -> one row per 30 s."""
        self.db, self.root, self.stride = db, root, series_stride

    def _ref(self, path):
        return self.db.reference(f"{self.root}/{path}")

    def upsert_event(self, ev):
        self._ref(f"faults/{ev['event_id']}").set(sanitize(ev))

    def upsert_sms(self, rec):
        self._ref(f"sms/{rec['sms_id']}").set(sanitize(rec))

    def write_run(self, result):
        meta = {"run_id": result.run_id, "scenario": result.scenario.sid, "name": result.scenario.name,
                "purpose": result.scenario.purpose, "data_source": "HIL_SIMULATED",
                "started_epoch": result.wall_start, "duration_s": result.scenario.duration_s,
                "dt_s": result.cfg.dt_s, "mitigation_enabled": result.scenario.mitigation_enabled,
                "validated_inputs": result.scenario.use_validated_inputs,
                "metrics": result.metrics, "series_stride": self.stride,
                "events": [e["event_id"] for e in result.events]}
        self._ref(f"runs/{result.run_id}/meta").set(sanitize(meta))
        df = result.df.iloc[:: self.stride]
        payload = {}
        for i, row in enumerate(df.to_dict("records")):
            payload[f"{i:04d}"] = row
        self._ref(f"runs/{result.run_id}/series").update(sanitize(payload))

    # ---- one-time schema bootstrap: never overwrites anything that already exists ----
    def init_schema(self, cfg_dict):
        created, kept = [], []

        def put_if_absent(path, value):
            ref = self._ref(path)
            if ref.get() is None:
                ref.set(sanitize(value))
                created.append(f"{self.root}/{path}")
            else:
                kept.append(f"{self.root}/{path}")

        put_if_absent("_schema", {"version": SCHEMA_VERSION, "description": __doc__.strip().split("\n\n")[0],
                                  "layout": __doc__.strip()})
        put_if_absent("config/thresholds", cfg_dict)
        put_if_absent("control", {"scenario_cmd": "idle", "pump_override": "AUTO",
                                  "solenoid_cmd_SIM": "CLOSED", "note": "dashboard -> HIL commands"})
        put_if_absent("status", {"state": "idle", "last_run_id": "none", "data_source": "HIL_SIMULATED"})
        return created, kept
