"""
STMaCS feature builder - turns live sensor samples into the 54 CORE features
that the safe forecaster expects.

The definitions below were recovered from, and verified against, the real
system 455 training data (every feature reproduced exactly - see
validate_features.py).

Inputs per sample (one every ~5 minutes)
----------------------------------------
  timestamp        datetime, in the SAME clock the training data used
                   (local logger time - keep it consistent!)
  buffer_top       tank top temperature, C       (caravan: s3)
  buffer_centre    tank middle temperature, C    (caravan: s2)
  buffer_bottom    tank bottom temperature, C    (caravan: s4)
  collector_temp   solar collector/loop temp, C  (caravan: s1)
  solar_flow       solar pump flow, in the SAME UNITS as the 455 data
  solar_pump       1 if the pump is commanded on, else 0
                   (pass the real pump state - do not derive it from flow:
                   in the 455 data the two agree only ~92% of the time)

Feature definitions
-------------------
  current          : the six inputs above
  stratification   : top-centre, centre-bottom, top-bottom
  collector minus  : collector_temp - buffer_top / centre / bottom
  lag_k            : the value k samples ago (k = 1, 3, 6, 12 -> 5, 15, 30,
                     60 minutes) for buffer_top, buffer_centre, buffer_bottom,
                     collector_temp, solar_flow
  delta_k          : current value - lag_k, for the four temperatures
  time features    : sin/cos of hour (with minutes and seconds, period 24 h),
                   day of week (Monday = 0, period 7) and
                   day of year (period 365.25)

History requirement
-------------------
Lags reach back 60 minutes, so the builder needs about an hour of samples
before it can produce features. After a power-up, or after a gap in data, it
reports "not ready" until enough history has accumulated again. The controller
must have a fallback for that period (for example a simple rule-based
decision).

Usage
-----
    from stmacs_features import FeatureBuilder

    fb = FeatureBuilder()
    fb.update(timestamp, buffer_top, buffer_centre, buffer_bottom,
              collector_temp, solar_flow, solar_pump)
    if fb.ready():
        row = fb.features()                  # dict with the 54 features
        result = forecaster.predict(row, hour=timestamp.hour)
"""

import bisect
import math
from collections import OrderedDict

STEP_MIN = 5
LAGS = (1, 3, 6, 12)
LAGGED_CHANNELS = ("buffer_top", "buffer_centre", "buffer_bottom",
                   "collector_temp", "solar_flow")
DELTA_CHANNELS = ("buffer_top", "buffer_centre", "buffer_bottom",
                  "collector_temp")
INPUTS = ("buffer_top", "buffer_centre", "buffer_bottom",
          "collector_temp", "solar_flow", "solar_pump")


class InsufficientHistory(Exception):
    """Raised when the history needed for the lag features is not available."""


def _to_seconds(ts):
    return ts.timestamp()


class FeatureBuilder:

    def __init__(self, tolerance_s=150, keep_minutes=180):
        # A lag counts as available if a sample exists within tolerance_s
        # (default 150 s = half a step) of the requested time.
        self.tolerance_s = tolerance_s
        self.keep_s = keep_minutes * 60
        self._t = []        # sample times, seconds
        self._v = []        # sample values (dict per sample)
        self._latest_ts = None

    # ------------------------------------------------------------------
    def update(self, timestamp, buffer_top, buffer_centre, buffer_bottom,
               collector_temp, solar_flow, solar_pump):
        t = _to_seconds(timestamp)
        if self._t and t <= self._t[-1]:
            raise ValueError("timestamps must be strictly increasing")
        self._t.append(t)
        self._v.append({
            "buffer_top": float(buffer_top),
            "buffer_centre": float(buffer_centre),
            "buffer_bottom": float(buffer_bottom),
            "collector_temp": float(collector_temp),
            "solar_flow": float(solar_flow),
            "solar_pump": float(solar_pump),
        })
        self._latest_ts = timestamp

        # Drop history older than keep_s
        cutoff = t - self.keep_s
        i = bisect.bisect_left(self._t, cutoff)
        if i > 500:
            del self._t[:i]
            del self._v[:i]

    # ------------------------------------------------------------------
    def _nearest(self, target_s):
        """Index of the sample nearest to target_s, or None if outside tolerance."""
        i = bisect.bisect_left(self._t, target_s)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(self._t)]
        if not candidates:
            return None
        j = min(candidates, key=lambda x: abs(self._t[x] - target_s))
        return j if abs(self._t[j] - target_s) <= self.tolerance_s else None

    def _lag_indices(self):
        t_now = self._t[-1]
        idx = {}
        for k in LAGS:
            idx[k] = self._nearest(t_now - k * STEP_MIN * 60)
        return idx

    def missing_lags(self):
        """List of lag steps (1, 3, 6, 12) that cannot be served yet."""
        if not self._t:
            return list(LAGS)
        idx = self._lag_indices()
        return [k for k in LAGS if idx[k] is None]

    def ready(self):
        return bool(self._t) and not self.missing_lags()

    # ------------------------------------------------------------------
    def features(self):
        if not self._t:
            raise InsufficientHistory("no samples yet")
        idx = self._lag_indices()
        missing = [k for k in LAGS if idx[k] is None]
        if missing:
            raise InsufficientHistory(
                f"missing history for lag(s) {missing} "
                f"(steps of {STEP_MIN} min); need about "
                f"{max(LAGS) * STEP_MIN} min of continuous samples")

        now = self._v[-1]
        ts = self._latest_ts
        f = OrderedDict()

        # current state
        for name in INPUTS:
            f[name] = now[name]

        # stratification and collector differences
        f["thermal_stratification_top_centre"] = (
            now["buffer_top"] - now["buffer_centre"])
        f["thermal_stratification_centre_bottom"] = (
            now["buffer_centre"] - now["buffer_bottom"])
        f["thermal_stratification_top_bottom"] = (
            now["buffer_top"] - now["buffer_bottom"])
        f["collector_minus_buffer_top"] = (
            now["collector_temp"] - now["buffer_top"])
        f["collector_minus_buffer_centre"] = (
            now["collector_temp"] - now["buffer_centre"])
        f["collector_minus_buffer_bottom"] = (
            now["collector_temp"] - now["buffer_bottom"])

        # time features
        hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
        f["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        f["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
        dow = ts.weekday()                       # Monday = 0
        f["day_of_week_sin"] = math.sin(2 * math.pi * dow / 7.0)
        f["day_of_week_cos"] = math.cos(2 * math.pi * dow / 7.0)
        doy = ts.timetuple().tm_yday
        f["day_of_year_sin"] = math.sin(2 * math.pi * doy / 365.25)
        f["day_of_year_cos"] = math.cos(2 * math.pi * doy / 365.25)

        # lags
        for ch in LAGGED_CHANNELS:
            for k in LAGS:
                f[f"{ch}_lag_{k}"] = self._v[idx[k]][ch]

        # deltas (current minus lag)
        for ch in DELTA_CHANNELS:
            for k in LAGS:
                f[f"{ch}_delta_{k}"] = now[ch] - self._v[idx[k]][ch]

        return dict(f)

    def reset(self):
        """Call after a power-down or when starting a fresh run."""
        self._t.clear()
        self._v.clear()
        self._latest_ts = None