"""
STMaCS - AI early-warning safety layer for Predictive Control.

What it is
----------
The trained models (feature builder -> SafeForecaster -> rule-based Controller, exactly as
in ai_closed_loop.py) wrapped so they can run INSIDE the Digital Twin demo day, next to the
day-ahead planner on the Predictive Control page.

How it combines with the planner
--------------------------------
The planner (twin physics + weather forecast) still decides WHEN to heat to meet the morning
and evening demand windows. This layer watches the live sensors every 5 simulated minutes,
forecasts the tank top (S3) 1 h and 2 h ahead with the pessimistic (Q10) models, and says
"heater ON" when the forecast is at risk of dropping below the comfort limit. In
predictive_control.decide_command the heater runs if EITHER the plan OR this layer wants it
(safety limit and manual override still come first). It never switches the heater off by
itself: it can only ADD heating that the plan did not schedule, so a forecast surprise
(e.g. forecast clear, actual overcast) is caught by the sensors.

Warm-up
-------
The feature builder needs 60 minutes of continuous 5-minute samples. Until then the layer
reports "warming up" and abstains (ready = False); the planner alone is in charge. The plain
thermostat fallback used by ai_closed_loop.py is deliberately NOT applied here, because it
would heat at 04:00 whenever S3 starts below 47 C and hide what the plan does.

Files needed next to this one (all in utils/): stmacs_features.py, stmacs_forecaster.py,
stmacs_controller.py, plus the trained models folder (default models/twin_v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:                                           # package layout used by the app
    from utils.stmacs_features import FeatureBuilder, InsufficientHistory
    from utils.stmacs_forecaster import SafeForecaster
    from utils.stmacs_controller import Controller, ControllerConfig, HeaterModel
except ImportError:                            # flat layout (scripts / tests)
    from stmacs_features import FeatureBuilder, InsufficientHistory
    from stmacs_forecaster import SafeForecaster
    from stmacs_controller import Controller, ControllerConfig, HeaterModel

SAMPLE_MIN = 5                  # the models were trained on 5-minute samples
EXTRA_CUSHION_2H_C = 2.0        # same extra margin on the 2 h forecast as ai_closed_loop.py
SENSOR_NOISE_C = 0.1            # what the models "see": 0.1 C noise and 0.1 C rounding


@dataclass
class AIOpinion:
    ready: bool                 # True once the forecast models are producing opinions
    want_on: bool               # the layer asks for the heater to be ON
    mode: str                   # "forecast", "warming up" or "fallback"
    reason: str
    forecast: dict = field(default_factory=dict)   # {"1 h": (pessimistic, threshold), ...}
    history_min: float = 0.0
    time: datetime | None = None

    def as_dict(self) -> dict:
        return {"ready": self.ready, "want_on": self.want_on, "mode": self.mode,
                "reason": self.reason, "forecast": self.forecast,
                "history_min": self.history_min, "time": self.time}


class AISafetyLayer:

    def __init__(self, model_dir, heater_power_w: float = 5000.0, volume_l: float = 300.0,
                 seed: int = 7):
        self.model_dir = Path(model_dir)
        self.forecaster = SafeForecaster(self.model_dir)      # raises if the folder is wrong
        for h in (1, 2):
            if h not in self.forecaster.horizons:
                raise ValueError(f"The model folder has no {h}-hour model.")
        self.heater_power_w = float(heater_power_w)
        self.volume_l = float(volume_l)
        self._seed = seed
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call at the start of every run (also done automatically if the clock goes back)."""
        self.fb = FeatureBuilder()
        self.ctrl = Controller(
            HeaterModel(power_kw=max(self.heater_power_w, 1.0) / 1000.0, volume_l=self.volume_l),
            ControllerConfig(extra_cushion={2: EXTRA_CUSHION_2H_C}))
        self._rng = np.random.default_rng(self._seed + 1000)
        self._first: datetime | None = None
        self._last_sample: datetime | None = None
        self.opinion = AIOpinion(False, False, "warming up", "waiting for the first sample")

    # ------------------------------------------------------------------
    def _predict(self, features, hour):
        try:                                   # newer forecaster (as used by ai_closed_loop.py)
            return self.forecaster.predict(features, hour=hour, horizons=(1, 2), expected=False)
        except TypeError:                      # older forecaster without those arguments
            return self.forecaster.predict(features, hour=hour)

    def _decide(self, now, top, forecast, centre):
        try:                                   # newer controller (accepts the middle sensor)
            return self.ctrl.step(now, top, forecast, buffer_centre=centre)
        except TypeError:                      # older controller
            return self.ctrl.step(now, top, forecast)

    # ------------------------------------------------------------------
    def step(self, now: datetime, s1: float, s2: float, s3: float, s4: float,
             flow_l_per_h: float, pump_on: bool) -> AIOpinion:
        """Call every simulated minute; the models only act on a 5-minute grid."""
        if self._last_sample is not None and now < self._last_sample:
            self.reset()                       # a new run started
        if (self._last_sample is not None
                and (now - self._last_sample).total_seconds() < SAMPLE_MIN * 60 - 1):
            return self.opinion                # between samples: keep the last opinion

        self._last_sample = now
        if self._first is None:
            self._first = now
        history_min = (now - self._first).total_seconds() / 60.0

        # what the AI sees is a noisy, rounded copy of the sensors
        top, centre, bottom, coll = (round(float(v) + self._rng.normal(0, SENSOR_NOISE_C), 1)
                                     for v in (s3, s2, s4, s1))
        self.fb.update(now, top, centre, bottom, coll, float(flow_l_per_h), int(bool(pump_on)))

        try:
            forecast = self._predict(self.fb.features(), now.hour)
        except InsufficientHistory:
            self.opinion = AIOpinion(
                False, False, "warming up",
                f"collecting history ({min(history_min, 60):.0f}/60 min); the planner is in charge",
                history_min=history_min, time=now)
            return self.opinion

        decision = self._decide(now, top, forecast, centre)
        shown = {}
        for h in (1, 2):
            r = forecast[h]
            thr = float(r["warning_threshold"]) + (EXTRA_CUSHION_2H_C if h == 2 else 0.0)
            shown[f"{h} h"] = (float(r["pessimistic_top"][0]), thr)
        self.opinion = AIOpinion(True, bool(decision.heater_on), decision.mode, decision.reason,
                                 shown, history_min, now)
        return self.opinion