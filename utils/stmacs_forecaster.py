"""
STMaCS safe forecaster - inference wrapper for the digital twin / controller.

Loads the models saved by train_final_models.py and turns a row of current
system features into forecasts and a "risk of unusable water" flag for each
horizon.

Example
-------
    from stmacs_forecaster import SafeForecaster

    fc = SafeForecaster(r"C:\\...\\STMaCS\\models\\stmacs_v1")
    result = fc.predict(feature_row, hour=20)   # decision hour 0-23

    for horizon, r in result.items():
        print(horizon, r["pessimistic_top"], r["expected_top"], r["risk"])

feature_row must contain every column listed in fc.features (the 54 CORE
features: tank temperatures, collector temperature, pump flow/state, time
features, and their lags/deltas), computed exactly as in
build_ml_datasets.py. It can be a pandas DataFrame (one or many rows) or a dict
(one row). `hour` is the decision hour of day (0-23): an int for one row, or an
array with one value per row.

Optional arguments: horizons=(1, 2) computes only those horizons, and
expected=False skips the median model (faster when only the safety forecast is
needed).

Output per horizon (arrays, one value per input row)
----------------------------------------------------
  pessimistic_top    safety forecast of buffer_top (Q10: the tank should be
                     at least this warm about 90% of the time)
  expected_top       median forecast of buffer_top (for dashboards)
  warning_threshold  usable_temp + validated margin + cushion (degrees C)
  risk               True if pessimistic_top < warning_threshold, i.e. the
                     controller should treat the water as at risk of being
                     unusable at that horizon
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class SafeForecaster:

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        config_file = self.model_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")

        self.config = json.loads(config_file.read_text())
        self.features = list(self.config["features"])
        self.horizons = list(self.config["horizons_h"])
        self.usable_temp = float(self.config["usable_temp_c"])

        self.bundles = {}
        for h in self.horizons:
            path = self.model_dir / self.config["bundles"][str(h)]["file"]
            self.bundles[h] = joblib.load(path)

    def predict(self, features, hour, horizons=None, expected=True):
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a dict or a pandas DataFrame")

        missing = [c for c in self.features if c not in features.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} required feature(s) missing, e.g. {missing[:5]}")

        hours = np.atleast_1d(np.asarray(hour)).astype(int)
        if len(hours) == 1 and len(features) > 1:
            hours = np.repeat(hours, len(features))
        if len(hours) != len(features):
            raise ValueError("hour must be one value or one value per row")
        if hours.min() < 0 or hours.max() > 23:
            raise ValueError("hour must be between 0 and 23")

        X = features[self.features]
        if X.isna().any().any():
            raise ValueError("feature row contains missing values (NaN)")

        current_top = features["buffer_top"].to_numpy()
        results = {}

        wanted = [h for h in (horizons if horizons is not None else self.bundles)
                  if h in self.bundles]
        for h in wanted:
            b = self.bundles[h]
            anchor = current_top + np.asarray(b["delta_by_hour"])[hours]
            pessimistic = anchor + b["q10_model"].predict(X)
            expected_top = anchor + b["q50_model"].predict(X) if expected else None
            threshold = b["usable_temp"] + b["margin"] + b["cushion"]
            results[h] = {
                "pessimistic_top": pessimistic,
                "expected_top": expected_top,
                "warning_threshold": threshold,
                "risk": pessimistic < threshold,
            }
        return results

    def summary(self):
        lines = [f"{self.config['name']}",
                 f"usable temperature: {self.usable_temp:.0f} C at buffer_top"]
        for h in self.horizons:
            b = self.config["bundles"][str(h)]
            lines.append(
                f"  {h:>2} h ahead: warn if pessimistic forecast < "
                f"{b['warning_threshold_c']:.2f} C "
                f"(margin {b['margin']:+.2f} + cushion {b['cushion']:+.2f})")
        return "\n".join(lines)