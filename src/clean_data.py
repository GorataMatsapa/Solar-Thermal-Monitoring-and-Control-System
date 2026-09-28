
"""
STMaCS SYSTEM 455 DATA CLEANING

Purpose:
    Create a scientifically defensible cleaned dataset for System 455.

Input:
    data/raw/E2025-08-27_2026-08-27.csv

Output:
    data/processed/system455_clean.csv

Important:
    - Does NOT use weather API data.
    - Does NOT invent missing measurements.
    - Does NOT delete rows because of unusual but potentially physical values.
    - Only applies explicitly defined cleaning rules.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "real" / "E2025-08-27_2026-08-27.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "system455_clean.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_COLUMN = "buffer_top"

# Temperature screening limits.
# These are deliberately conservative screening rules.
BUFFER_MIN = 0.0
BUFFER_MAX = 100.0

COLLECTOR_MIN = -20.0
COLLECTOR_MAX = 180.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(df, candidates):
    """
    Find a column using exact names first, then case-insensitive matching.
    """
    columns = list(df.columns)

    # Exact match
    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Case-insensitive match
    lower_map = {str(col).strip().lower(): col for col in columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def numeric(series):
    """
    Convert a column to numeric safely.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STMaCS SYSTEM 455 DATA CLEANING")
print("=" * 80)

print("Loading raw dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find the raw dataset:\n{INPUT_FILE}\n\n"
        "Check that the CSV is inside data/raw/."
    )


# -----------------------------------------------------------------------------
# IMPORTANT:
# The university dataset uses ';' as its CSV delimiter.
# -----------------------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    low_memory=False
)

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# =============================================================================
# NORMALISE COLUMN NAMES
# =============================================================================

print("\n" + "=" * 80)
print("NORMALISING COLUMN NAMES")
print("=" * 80)

df.columns = [str(col).strip() for col in df.columns]

print("Column names stripped of leading/trailing whitespace.")


# =============================================================================
# IDENTIFY DATE AND TIME
# =============================================================================

print("\n" + "=" * 80)
print("CREATING TIMESTAMP")
print("=" * 80)

date_col = find_column(
    df,
    [
        "Date",
        "date",
        "DATE"
    ]
)

time_col = find_column(
    df,
    [
        "Time",
        "time",
        "TIME"
    ]
)

if date_col is None or time_col is None:

    print("\nAvailable columns:")
    for col in df.columns[:20]:
        print(f"  {col!r}")

    raise ValueError(
        "\nDate/Time columns could not be identified."
    )

print(f"Date column: {date_col}")
print(f"Time column: {time_col}")

df["timestamp"] = pd.to_datetime(
    df[date_col].astype(str).str.strip()
    + " "
    + df[time_col].astype(str).str.strip(),
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps:,}")

if invalid_timestamps > 0:
    print("Removing rows with invalid timestamps...")

    df = df.dropna(subset=["timestamp"]).copy()


# =============================================================================
# IDENTIFY SYSTEM 455 VARIABLES
# =============================================================================

print("\n" + "=" * 80)
print("IDENTIFYING SYSTEM 455 VARIABLES")
print("=" * 80)


column_map = {

    "collector_temp": [
        "1/Ana.1 - 1: 455_collector_temp"
    ],

    "buffer_bottom": [
        "1/Ana.2 - 2: 455_temp_buffer_bottom"
    ],

    "buffer_centre": [
        "1/Ana.3 - 3: 455_temp_buffer_centre"
    ],

    "buffer_top": [
        "1/Ana.4 - 4: 455_temp_buffer_top"
    ],

    "solar_radiation": [
        "1/Ana.5 - 5: 455_solar_radiation"
    ],

    "ambient_temp": [
        "1/Ana.6 - 6: 455_ambient_temp"
    ],

    "solar_energy": [
        "1/Ana.8 - 1: 455_HM_solar_energy"
    ],

    "solar_flow": [
        "1/Ana.9 - 2: 455_HM_solar_flow"
    ],

    "solar_power": [
        "1/Ana.10 - 3: 455_HM_solar_power"
    ],

    "solar_temp_hot": [
        "1/Ana.11 - 4: 455_HM_solar_temp_hot"
    ],

    "solar_temp_cold": [
        "1/Ana.12 - 5: 455_HM_solar_temp_cold"
    ],

    "hotwater_energy": [
        "1/Ana.14 - 8: 455_HM_hotwater_energy"
    ],

    "hotwater_volume": [
        "1/Ana.15 - 9: 455_HM_hotwater_volume"
    ],

    "hotwater_flow": [
        "1/Ana.16 - 10: 455_HM_hotwater_flow"
    ],

    "hotwater_power": [
        "1/Ana.17 - 11: 455_HM_hotwater_power"
    ],

    "hotwater_temp_hot": [
        "1/Ana.18 - 12: 455_HM_hotwater_temp_hot"
    ],

    "hotwater_temp_cold": [
        "1/Ana.19 - 13: 455_HM_hotwate_temp_cold"
    ],

    "pwm": [
        "1/Ana.22 - 10: 455_PWM"
    ],

    "solar_pump": [
        "1/Dig.1 - 1: 455_solar_pump"
    ],

    "electric_heater": [
        "1/Dig.3 - 3: 455_e-heater"
    ]
}


resolved = {}

for name, candidates in column_map.items():

    found = find_column(df, candidates)

    if found is None:
        print(f"  [NOT FOUND] {name}")
    else:
        resolved[name] = found
        print(f"  [FOUND] {name:<20} -> {found}")


# Required variables for the ML problem
required_variables = [
    "collector_temp",
    "buffer_bottom",
    "buffer_centre",
    "buffer_top",
    "solar_radiation",
    "ambient_temp",
    "solar_flow",
    "solar_power",
    "solar_temp_hot",
    "solar_temp_cold",
    "hotwater_flow",
    "hotwater_power",
    "hotwater_temp_hot",
    "hotwater_temp_cold",
    "solar_pump",
    "electric_heater"
]


missing_required = [
    variable
    for variable in required_variables
    if variable not in resolved
]

if missing_required:

    raise ValueError(
        "\nThe following required System 455 variables were not found:\n"
        + "\n".join(f"  - {x}" for x in missing_required)
    )


# =============================================================================
# CREATE CLEAN DATAFRAME
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING SYSTEM 455 DATASET")
print("=" * 80)


clean = pd.DataFrame()

clean["timestamp"] = df["timestamp"]

for name, original_column in resolved.items():

    clean[name] = numeric(df[original_column])


# =============================================================================
# SORT AND REMOVE DUPLICATE TIMESTAMPS
# =============================================================================

print("\nSorting chronologically...")

clean = clean.sort_values("timestamp").reset_index(drop=True)

duplicate_count = clean["timestamp"].duplicated().sum()

print(f"Duplicate timestamps: {duplicate_count:,}")

if duplicate_count > 0:

    print("Keeping the first occurrence of duplicate timestamps.")

    clean = clean.drop_duplicates(
        subset="timestamp",
        keep="first"
    ).reset_index(drop=True)


# =============================================================================
# CREATE TIME FEATURES
# =============================================================================

print("\nCreating time features...")

clean["hour"] = clean["timestamp"].dt.hour

clean["minute"] = clean["timestamp"].dt.minute

clean["day_of_year"] = clean["timestamp"].dt.dayofyear

clean["day_of_week"] = clean["timestamp"].dt.dayofweek

clean["month"] = clean["timestamp"].dt.month

clean["is_weekend"] = (
    clean["day_of_week"] >= 5
).astype(int)


# Cyclical time features
clean["hour_sin"] = np.sin(
    2 * np.pi * clean["hour"] / 24
)

clean["hour_cos"] = np.cos(
    2 * np.pi * clean["hour"] / 24
)

clean["day_of_year_sin"] = np.sin(
    2 * np.pi * clean["day_of_year"] / 365.25
)

clean["day_of_year_cos"] = np.cos(
    2 * np.pi * clean["day_of_year"] / 365.25
)


# =============================================================================
# QUALITY FLAGS
# =============================================================================

print("\n" + "=" * 80)
print("CREATING QUALITY FLAGS")
print("=" * 80)


# Buffer bottom below zero
clean["flag_buffer_bottom_below_0"] = (
    clean["buffer_bottom"] < BUFFER_MIN
)


# Collector outside broad screening range
clean["flag_collector_extreme"] = (
    (clean["collector_temp"] < COLLECTOR_MIN)
    |
    (clean["collector_temp"] > COLLECTOR_MAX)
)


# Buffer temperatures outside broad physical screening range
clean["flag_buffer_bottom_extreme"] = (
    (clean["buffer_bottom"] < BUFFER_MIN)
    |
    (clean["buffer_bottom"] > BUFFER_MAX)
)

clean["flag_buffer_centre_extreme"] = (
    (clean["buffer_centre"] < BUFFER_MIN)
    |
    (clean["buffer_centre"] > BUFFER_MAX)
)

clean["flag_buffer_top_extreme"] = (
    (clean["buffer_top"] < BUFFER_MIN)
    |
    (clean["buffer_top"] > BUFFER_MAX)
)


# =============================================================================
# DO NOT AUTOMATICALLY DELETE SUSPICIOUS TEMPERATURES
# =============================================================================

print("\nSuspicious values are being FLAGGED, not deleted.")

print(
    f"Buffer bottom < 0°C: "
    f"{clean['flag_buffer_bottom_below_0'].sum():,}"
)

print(
    f"Extreme collector values: "
    f"{clean['flag_collector_extreme'].sum():,}"
)


# =============================================================================
# MISSING DATA REPORT
# =============================================================================

print("\n" + "=" * 80)
print("MISSING DATA SUMMARY")
print("=" * 80)

measurement_columns = list(resolved.keys())

for column in measurement_columns:

    missing = clean[column].isna().sum()

    completeness = (
        100 * (1 - missing / len(clean))
    )

    print(
        f"{column:<25}"
        f"Missing: {missing:>8,}"
        f"   Complete: {completeness:>7.2f}%"
    )


# =============================================================================
# CREATE FUTURE TARGET
# =============================================================================

print("\n" + "=" * 80)
print("CREATING FUTURE TARGETS")
print("=" * 80)

# These targets are created using the actual chronological rows.
#
# IMPORTANT:
# We do NOT assume that every row is exactly 5 minutes apart.
# The actual timestamps are retained so that later modelling can
# account for irregular intervals/gaps.

clean["target_buffer_top_5min"] = (
    clean["buffer_top"].shift(-1)
)

clean["target_buffer_top_15min"] = (
    clean["buffer_top"].shift(-3)
)

clean["target_buffer_top_30min"] = (
    clean["buffer_top"].shift(-6)
)

clean["target_buffer_top_60min"] = (
    clean["buffer_top"].shift(-12)
)


# =============================================================================
# FINAL DATASET SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

print(f"Rows:    {len(clean):,}")
print(f"Columns: {len(clean.columns):,}")

print(
    f"\nFirst timestamp: "
    f"{clean['timestamp'].min()}"
)

print(
    f"Last timestamp:  "
    f"{clean['timestamp'].max()}"
)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

clean.to_csv(
    OUTPUT_FILE,
    index=False
)


# =============================================================================
# VERIFY OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(f"Clean dataset saved to:")

print(
    f"  {OUTPUT_FILE}"
)

print("\nVerification:")

if OUTPUT_FILE.exists():

    file_size_mb = (
        OUTPUT_FILE.stat().st_size / (1024 ** 2)
    )

    print("  [OK] File created successfully.")
    print(f"  File size: {file_size_mb:.2f} MB")

else:

    raise RuntimeError(
        "Output file was not created."
    )


print("\n" + "=" * 80)
print("CLEANING COMPLETE")
print("=" * 80)

print(
    """
No raw data was overwritten.

No suspicious temperature measurements were
automatically replaced.

No weather data was added.

No interpolation was performed.

The cleaned System 455 dataset is now ready
for the next preprocessing/feature-engineering stage.
"""
)

