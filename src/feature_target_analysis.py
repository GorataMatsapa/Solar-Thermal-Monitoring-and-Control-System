import pandas as pd
import numpy as np

# ============================================================
# STMaCS - SYSTEM 455 FEATURE / TARGET FEASIBILITY ANALYSIS
# ============================================================

DATA_PATH = "data/real/E2025-08-27_2026-08-27.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df):,}")

# ------------------------------------------------------------
# 1. HANDLE ERROR CODE
# ------------------------------------------------------------

# Electrical/data logger error value
df.replace(3276.7, np.nan, inplace=True)


# ------------------------------------------------------------
# 2. CREATE TIMESTAMP
# ------------------------------------------------------------

print("\nCreating combined timestamp from Date + Time...")

if "Date" not in df.columns or "Time" not in df.columns:
    raise ValueError("Expected 'Date' and 'Time' columns were not found.")

df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

df = df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------
# 3. SYSTEM 455 COLUMN DEFINITIONS
# ------------------------------------------------------------

features = {
    "collector_temp": "1/Ana.1 - 1: 455_collector_temp",
    "buffer_bottom": "1/Ana.2 - 2: 455_temp_buffer_bottom",
    "buffer_centre": "1/Ana.3 - 3: 455_temp_buffer_centre",
    "buffer_top": "1/Ana.4 - 4: 455_temp_buffer_top",
    "solar_radiation": "1/Ana.5 - 5: 455_solar_radiation",
    "solar_energy": "1/Ana.8 - 1: 455_HM_solar_energy",
    "solar_flow": "1/Ana.9 - 2: 455_HM_solar_flow",
    "solar_power": "1/Ana.10 - 3: 455_HM_solar_power",
    "solar_temp_hot": "1/Ana.11 - 4: 455_HM_solar_temp_hot",
    "solar_temp_cold": "1/Ana.12 - 5: 455_HM_solar_temp_cold",
    "hotwater_energy": "1/Ana.14 - 8: 455_HM_hotwater_energy",
    "hotwater_volume": "1/Ana.15 - 9: 455_HM_hotwater_volume",
    "hotwater_flow": "1/Ana.16 - 10: 455_HM_hotwater_flow",
    "hotwater_power": "1/Ana.17 - 11: 455_HM_hotwater_power",
    "hotwater_temp_hot": "1/Ana.18 - 12: 455_HM_hotwater_temp_hot",
    "hotwater_temp_cold": "1/Ana.19 - 13: 455_HM_hotwate_temp_cold",
    "pwm": "1/Ana.22 - 10: 455_PWM",
    "solar_pump": "1/Dig.1 - 1: 455_solar_pump",
    "electric_heater": "1/Dig.3 - 3: 455_e-heater",
}

# Verify columns exist
print("\nChecking System 455 columns...")

missing_columns = []

for name, column in features.items():
    if column in df.columns:
        print(f"  [FOUND] {name:<20} -> {column}")
    else:
        print(f"  [MISSING] {name:<20} -> {column}")
        missing_columns.append(column)

if missing_columns:
    raise ValueError(
        "\nSome expected System 455 columns were not found. "
        "Check the column names in the CSV."
    )


# ------------------------------------------------------------
# 4. SYSTEM 455 DATA COMPLETENESS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("              SYSTEM 455 DATA COMPLETENESS")
print("=" * 80)

print(
    f"{'Variable':<25}"
    f"{'Valid':>12}"
    f"{'Missing':>12}"
    f"{'Complete %':>14}"
)

print("-" * 80)

for name, column in features.items():

    valid = df[column].notna().sum()
    missing = df[column].isna().sum()
    completion = (valid / len(df)) * 100

    print(
        f"{name:<25}"
        f"{valid:>12,}"
        f"{missing:>12,}"
        f"{completion:>13.2f}%"
    )


# ------------------------------------------------------------
# 5. NUMERIC STATISTICS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("              SYSTEM 455 NUMERIC STATISTICS")
print("=" * 80)

print(
    f"{'Variable':<25}"
    f"{'Min':>12}"
    f"{'P1':>12}"
    f"{'Median':>12}"
    f"{'P99':>12}"
    f"{'Max':>12}"
)

print("-" * 80)

for name, column in features.items():

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        continue

    print(
        f"{name:<25}"
        f"{series.min():>12.2f}"
        f"{series.quantile(0.01):>12.2f}"
        f"{series.median():>12.2f}"
        f"{series.quantile(0.99):>12.2f}"
        f"{series.max():>12.2f}"
    )


# ------------------------------------------------------------
# 6. OPERATING STATE ANALYSIS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("              SYSTEM 455 OPERATING STATE ANALYSIS")
print("=" * 80)

# Pump
pump = pd.to_numeric(df[features["solar_pump"]], errors="coerce")

pump_valid = pump.dropna()

if len(pump_valid) > 0:

    pump_on = (pump_valid > 0).sum()
    pump_off = (pump_valid == 0).sum()

    print("\nSOLAR PUMP")
    print(f"  ON records:  {pump_on:,} ({pump_on / len(pump_valid) * 100:.2f}%)")
    print(f"  OFF records: {pump_off:,} ({pump_off / len(pump_valid) * 100:.2f}%)")


# Electric heater
heater = pd.to_numeric(df[features["electric_heater"]], errors="coerce")

heater_valid = heater.dropna()

if len(heater_valid) > 0:

    heater_on = (heater_valid > 0).sum()
    heater_off = (heater_valid == 0).sum()

    print("\nELECTRIC HEATER")
    print(
        f"  ON records:  {heater_on:,} "
        f"({heater_on / len(heater_valid) * 100:.2f}%)"
    )
    print(
        f"  OFF records: {heater_off:,} "
        f"({heater_off / len(heater_valid) * 100:.2f}%)"
    )


# ------------------------------------------------------------
# 7. CREATE TIME FEATURES
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                 TIME FEATURES")
print("=" * 80)

df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["day_of_week"] = df["timestamp"].dt.dayofweek

print("Created:")
print("  - hour")
print("  - day_of_year")
print("  - day_of_week")


# ------------------------------------------------------------
# 8. VERIFY ACTUAL SAMPLING INTERVAL
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                 SAMPLING INTERVAL")
print("=" * 80)

time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()

# Ignore extremely large gaps for determining normal sampling
normal_diffs = time_diffs[time_diffs <= 3600]

if len(normal_diffs) > 0:

    dominant_interval = normal_diffs.mode().iloc[0]

    print(f"Dominant interval: {dominant_interval:.0f} seconds")
    print(f"Dominant interval: {dominant_interval / 60:.2f} minutes")
    print(f"Median interval:   {normal_diffs.median():.0f} seconds")
    print(f"Minimum interval:  {normal_diffs.min():.0f} seconds")
    print(f"Maximum normal interval: {normal_diffs.max():.0f} seconds")


# ------------------------------------------------------------
# 9. CREATE FUTURE TEMPERATURE TARGETS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("              FUTURE TARGET FEASIBILITY")
print("=" * 80)

#
# Because the data is approximately every 5 minutes,
# these correspond approximately to:
#
# +1 = 5 minutes
# +3 = 15 minutes
# +6 = 30 minutes
# +12 = 60 minutes
#

target_source = features["buffer_top"]

df["target_5min"] = df[target_source].shift(-1)
df["target_15min"] = df[target_source].shift(-3)
df["target_30min"] = df[target_source].shift(-6)
df["target_60min"] = df[target_source].shift(-12)

targets = {
    "5 min": "target_5min",
    "15 min": "target_15min",
    "30 min": "target_30min",
    "60 min": "target_60min",
}

print("\nTarget variable:")
print("  Buffer Top Temperature")

print("\nAvailable target samples:")

for horizon, column in targets.items():

    valid = df[column].notna().sum()
    percentage = valid / len(df) * 100

    print(
        f"  {horizon:<6}: "
        f"{valid:>10,} valid samples "
        f"({percentage:.2f}%)"
    )


# ------------------------------------------------------------
# 10. TEMPERATURE CHANGE ANALYSIS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("            FUTURE TEMPERATURE CHANGE")
print("=" * 80)

current_temp = pd.to_numeric(
    df[target_source],
    errors="coerce"
)

horizons = {
    "5 min": 1,
    "15 min": 3,
    "30 min": 6,
    "60 min": 12,
}

print(
    f"{'Horizon':<12}"
    f"{'Mean ΔT':>15}"
    f"{'Std ΔT':>15}"
    f"{'Mean |ΔT|':>15}"
    f"{'Max |ΔT|':>15}"
)

print("-" * 72)

for horizon, steps in horizons.items():

    future = current_temp.shift(-steps)

    delta = future - current_temp

    delta = delta.dropna()

    if len(delta) == 0:
        continue

    print(
        f"{horizon:<12}"
        f"{delta.mean():>15.4f}"
        f"{delta.std():>15.4f}"
        f"{delta.abs().mean():>15.4f}"
        f"{delta.abs().max():>15.4f}"
    )


# ------------------------------------------------------------
# 11. BUILD FEATURE/TARGET DATASET
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("             FEATURE / TARGET FEASIBILITY")
print("=" * 80)

candidate_features = [
    "collector_temp",
    "buffer_bottom",
    "buffer_centre",
    "buffer_top",
    "solar_radiation",
    "solar_flow",
    "solar_power",
    "solar_temp_hot",
    "solar_temp_cold",
    "hotwater_flow",
    "hotwater_power",
    "hotwater_temp_hot",
    "hotwater_temp_cold",
    "pwm",
    "solar_pump",
    "electric_heater",
    "hour",
    "day_of_year",
    "day_of_week",
]

feature_columns = [features[x] for x in candidate_features if x in features]

model_columns = feature_columns + [
    "hour",
    "day_of_year",
    "day_of_week",
    "target_5min",
    "target_15min",
    "target_30min",
    "target_60min",
]

model_df = df[model_columns].copy()

print(f"Candidate input features: {len(candidate_features)}")

print("\nComplete cases for each prediction horizon:")

for horizon, target in targets.items():

    cols_for_target = feature_columns + [
        "hour",
        "day_of_year",
        "day_of_week",
        target,
    ]

    complete_rows = model_df[cols_for_target].dropna()

    print(
        f"  {horizon:<6}: "
        f"{len(complete_rows):>10,} complete samples"
    )


# ------------------------------------------------------------
# 12. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("          CORRELATION WITH 60-MIN FUTURE TEMPERATURE")
print("=" * 80)

correlation_df = model_df[
    feature_columns
    + ["hour", "day_of_year", "day_of_week", "target_60min"]
].copy()

correlation_df = correlation_df.apply(
    pd.to_numeric,
    errors="coerce"
)

correlations = (
    correlation_df.corr()["target_60min"]
    .drop("target_60min")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

print(
    f"{'Feature':<45}"
    f"{'Correlation':>15}"
)

print("-" * 62)

for feature, correlation in correlations.items():

    print(
        f"{feature:<45}"
        f"{correlation:>15.4f}"
    )


# ------------------------------------------------------------
# 13. SIMPLE DATA QUALITY FLAGS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                 DATA QUALITY FLAGS")
print("=" * 80)

# These are screening flags only.
# Nothing is deleted or modified.

quality_checks = {
    "buffer_bottom_below_0C":
        (pd.to_numeric(df[features["buffer_bottom"]], errors="coerce") < 0).sum(),

    "buffer_top_above_100C":
        (pd.to_numeric(df[features["buffer_top"]], errors="coerce") > 100).sum(),

    "buffer_centre_above_100C":
        (pd.to_numeric(df[features["buffer_centre"]], errors="coerce") > 100).sum(),

    "collector_above_180C":
        (pd.to_numeric(df[features["collector_temp"]], errors="coerce") > 180).sum(),

    "solar_radiation_negative":
        (pd.to_numeric(df[features["solar_radiation"]], errors="coerce") < 0).sum(),

    "solar_flow_negative":
        (pd.to_numeric(df[features["solar_flow"]], errors="coerce") < 0).sum(),

    "hotwater_flow_negative":
        (pd.to_numeric(df[features["hotwater_flow"]], errors="coerce") < 0).sum(),
}

for check, count in quality_checks.items():

    status = "FLAGGED" if count > 0 else "OK"

    print(f"  {status:<8} {check:<35}: {count:,}")


# ------------------------------------------------------------
# 14. FINAL ASSESSMENT
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                    PRELIMINARY ASSESSMENT")
print("=" * 80)

print("""
This analysis DOES NOT clean, delete, or modify suspicious measurements.

It evaluates whether System 455 contains enough usable information
to construct a supervised time-series prediction problem.

The main research target is:

    Future Buffer Top Temperature

Candidate prediction horizons:

    5 minutes
    15 minutes
    30 minutes
    60 minutes

Candidate predictors include:

    - Collector temperature
    - Buffer temperatures
    - Solar radiation
    - Solar thermal flow
    - Solar thermal power
    - Hot-water flow
    - Hot-water power
    - Heat-transfer temperatures
    - PWM
    - Solar pump state
    - Electric heater state
    - Time-of-day features
    - Seasonal/time features

The results will determine the next stage:

    1. Data cleaning
    2. Feature engineering
    3. Chronological train/test splitting
    4. Baseline ML model
    5. Model evaluation
""")

print("=" * 80)
print("                    ANALYSIS COMPLETE")
print("=" * 80)