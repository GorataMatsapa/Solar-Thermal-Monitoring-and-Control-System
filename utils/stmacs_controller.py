"""
STMaCS predictive heater controller.

Decides, every few minutes, whether the backup heater should be ON or OFF so
that the tank top stays usable (>= usable_temp) with as little heating as
possible. It uses the safe forecaster's pessimistic forecasts, a few simple
rules, and a heater physics helper. The controller itself is rule-based, not
learned.

Heater physics (mixed-tank approximation)
-----------------------------------------
  energy to warm the whole tank by 1 C = volume_kg * 4.186 kJ/(kg K) / 3600
  For 300 L this is about 0.349 kWh per C, so a 5 kW element warms a fully
  mixed tank by about 14 C per hour (about 42 minutes per 10 C), ignoring
  losses. If the element sits high in a stratified tank, the TOP warms faster
  than this, so this is a conservative (slow) estimate for the top.

Decision rules (checked in this order)
--------------------------------------
  1. HARD FLOOR     : top < usable_temp + floor_margin           -> ON
  2. FORECAST RISK  : at any decision horizon (default 1 h and 2 h) the
                      pessimistic forecast is below its warning threshold
                      (validated margin + cushion)                -> ON
  3. HOLD           : if the heater is ON, keep it ON until the top is at
                      least usable_temp + off_buffer AND every decision
                      horizon is clear by `hysteresis` degrees   -> then OFF
  4. otherwise      : OFF
  ELEMENT LIMIT     : if the caller passes the middle-sensor reading and it is at
                      or above middle_limit_c, the heater is forced OFF (this
                      beats the minimum on-time).
  Minimum ON and OFF times (default 10 min each) stop rapid switching. The
  hard floor overrides the minimum OFF time.

Fallback
--------
When no forecast is available (for example during the first hour after
power-up, while the feature builder is still collecting history), the
controller uses a plain thermostat with hysteresis.

Usage with the rest of the system
---------------------------------
    fb   = FeatureBuilder()
    fc   = SafeForecaster(r"...\\models\\stmacs_v2")
    ctrl = Controller(HeaterModel(power_kw=5.0, volume_l=300.0))

    # every ~5 minutes:
    fb.update(now, top, centre, bottom, collector, flow, pump_state)
    try:
        forecast = fc.predict(fb.features(), hour=now.hour)
    except InsufficientHistory:
        forecast = None
    decision = ctrl.step(now, top, forecast)
    heater_on = decision.heater_on        # send this to the twin / heater

Call ctrl.reset() and fb.reset() at power-up.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


# =============================================================================
# HEATER PHYSICS
# =============================================================================

@dataclass
class HeaterModel:
    power_kw: float = 5.0
    volume_l: float = 300.0
    specific_heat_kj_per_kg_k: float = 4.186
    efficiency: float = 1.0           # electric resistance element

    @property
    def kwh_per_degree(self):
        """Energy to warm the whole tank by 1 C (kWh)."""
        return self.volume_l * self.specific_heat_kj_per_kg_k / 3600.0

    @property
    def rise_rate_c_per_hour(self):
        """Mean tank temperature rise with the heater on, ignoring losses."""
        return self.power_kw * self.efficiency / self.kwh_per_degree

    def minutes_for_rise(self, delta_c):
        if delta_c <= 0:
            return 0.0
        return delta_c / self.rise_rate_c_per_hour * 60.0

    def energy_kwh_for_rise(self, delta_c):
        if delta_c <= 0:
            return 0.0
        return delta_c * self.kwh_per_degree / self.efficiency

    def temperature_rise(self, minutes):
        """Mean tank temperature rise after running for `minutes`."""
        return self.rise_rate_c_per_hour * minutes / 60.0


# =============================================================================
# CONTROLLER
# =============================================================================

@dataclass
class ControllerConfig:
    usable_temp: float = 45.0
    floor_margin: float = 1.0       # hard floor = usable_temp + floor_margin
    off_buffer: float = 3.0         # may stop once top >= usable_temp + this
    hysteresis: float = 1.5         # forecasts must clear threshold by this
    min_on_min: float = 10.0
    min_off_min: float = 10.0
    decision_horizons: tuple = (1, 2)
    # extra cushion (C) added on top of the forecaster's own warning threshold
    extra_cushion: Dict[int, float] = field(default_factory=lambda: {2: 2.0})
    fallback_on_below: float = 47.0
    fallback_off_at: float = 49.0
    middle_limit_c: float = 70.0    # heater forced OFF if the middle sensor (S2) reaches this


@dataclass
class Decision:
    heater_on: bool
    mode: str                         # "forecast" or "fallback"
    reason: str
    estimated_minutes: Optional[float] = None
    switched: bool = False
    detail: Dict = field(default_factory=dict)


class Controller:

    def __init__(self, heater: HeaterModel = None, config: ControllerConfig = None):
        self.heater = heater or HeaterModel()
        self.cfg = config or ControllerConfig()
        self.reset()

    def reset(self):
        self.heater_on = False
        self._last_switch: Optional[datetime] = None

    # ------------------------------------------------------------------
    def _minutes_since_switch(self, now):
        if self._last_switch is None:
            return float("inf")
        return (now - self._last_switch).total_seconds() / 60.0

    def _usable_forecast_horizons(self, forecast):
        if not forecast:
            return []
        return [h for h in self.cfg.decision_horizons if h in forecast]

    # ------------------------------------------------------------------
    def step(self, now: datetime, buffer_top: float, forecast=None,
             buffer_centre: Optional[float] = None) -> Decision:
        cfg = self.cfg
        floor = cfg.usable_temp + cfg.floor_margin
        horizons = self._usable_forecast_horizons(forecast)

        detail = {}
        emergency = False

        if not horizons:
            # ---------------- fallback thermostat ----------------
            mode = "fallback"
            if buffer_top < cfg.fallback_on_below:
                want, reason = True, (f"fallback: top {buffer_top:.1f} C below "
                                      f"{cfg.fallback_on_below:.0f} C")
            elif buffer_top >= cfg.fallback_off_at:
                want, reason = False, (f"fallback: top {buffer_top:.1f} C at or "
                                       f"above {cfg.fallback_off_at:.0f} C")
            else:
                want = self.heater_on
                reason = "fallback: inside hysteresis band, keeping state"
            emergency = buffer_top < floor
        else:
            # ---------------- forecast-based rules ----------------
            mode = "forecast"
            at_risk = []
            all_clear = True
            for h in horizons:
                r = forecast[h]
                pess = float(r["pessimistic_top"][0])
                thr = float(r["warning_threshold"]) + cfg.extra_cushion.get(h, 0.0)
                detail[f"{h}h"] = {"pessimistic_top": round(pess, 2),
                                   "threshold": round(thr, 2)}
                if pess < thr:
                    at_risk.append(h)
                if pess < thr + cfg.hysteresis:
                    all_clear = False
            all_clear = all_clear and buffer_top >= cfg.usable_temp + cfg.off_buffer

            if buffer_top < floor:
                want, reason = True, (f"hard floor: top {buffer_top:.1f} C below "
                                      f"{floor:.1f} C")
                emergency = True
            elif at_risk:
                want, reason = True, ("forecast risk at "
                                      + " and ".join(f"{h} h" for h in at_risk))
            elif self.heater_on and not all_clear:
                want, reason = True, "holding until forecasts clear safely"
            else:
                want, reason = False, "no risk forecast"

        # ---------------- element protection (S2 limit) ----------------
        over_limit = buffer_centre is not None and buffer_centre >= cfg.middle_limit_c
        if over_limit and want:
            want = False
            reason = (f"S2 {buffer_centre:.0f} C reached the {cfg.middle_limit_c:.0f} C "
                      "element limit")

        # ---------------- minimum on/off times ----------------
        since = self._minutes_since_switch(now)
        if over_limit:
            pass                                    # safety: switching off may not wait
        elif want != self.heater_on:
            if self.heater_on and since < cfg.min_on_min:
                want = True
                reason += f" (min on-time, {since:.0f}/{cfg.min_on_min:.0f} min)"
            elif (not self.heater_on) and since < cfg.min_off_min and not emergency:
                want = False
                reason += f" (min off-time, {since:.0f}/{cfg.min_off_min:.0f} min)"

        switched = want != self.heater_on
        if switched:
            self._last_switch = now
        self.heater_on = want

        estimated = None
        if want:
            deficit = (cfg.usable_temp + cfg.off_buffer) - buffer_top
            estimated = round(self.heater.minutes_for_rise(deficit), 1)

        return Decision(heater_on=want, mode=mode, reason=reason,
                        estimated_minutes=estimated, switched=switched,
                        detail=detail)


# =============================================================================
# DEMO (logic test with a toy tank - not a realistic simulation)
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from datetime import timedelta

    heater = HeaterModel(power_kw=5.0, volume_l=300.0)
    print(f"Heater: {heater.power_kw} kW on {heater.volume_l:.0f} L")
    print(f"  {heater.kwh_per_degree:.3f} kWh per C | "
          f"{heater.rise_rate_c_per_hour:.1f} C/hour | "
          f"{heater.minutes_for_rise(10):.0f} min and "
          f"{heater.energy_kwh_for_rise(10):.1f} kWh per 10 C\n")

    ctrl = Controller(heater)
    now = datetime(2026, 7, 10, 18, 0)
    top = 62.0
    trace = []
    for step in range(60):
        # toy forecast: tank loses 3 C in 1 h and 6 C in 2 h from where it is now
        fc = {
            1: {"pessimistic_top": np.array([top - 3.0]), "warning_threshold": 46.25},
            2: {"pessimistic_top": np.array([top - 6.0]), "warning_threshold": 48.75},
        }
        d = ctrl.step(now, top, fc)
        trace.append((now.strftime("%H:%M"), round(top, 1), d.heater_on,
                      d.reason[:48]))
        # toy tank: -0.6 C per step off, +1.19 C per step on (5 kW, 300 L)
        top += heater.temperature_rise(5) if d.heater_on else -0.6
        now += timedelta(minutes=5)

    last = None
    for t, temp, on, why in trace:
        if on != last:
            print(f"{t}  top {temp:5.1f} C  heater {'ON ' if on else 'OFF'}  {why}")
            last = on
    print(f"\nON for {sum(1 for x in trace if x[2])} of {len(trace)} steps")