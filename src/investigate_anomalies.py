import pandas as pd
import numpy as np

# ============================================================
# STMaCS - SYSTEM 455 ANOMALY INVESTIGATION
# ============================================================
#
# Purpose:
#   Investigate suspicious measurements in System 455
#   WITHOUT deleting or modifying the original data.
#
# Current focus:
#   1. Buffer bottom temperatures below 0°C
#   2. Sudden temperature changes
#   3. Suspicious thermal measurements
#   4. Context surrounding flagged observations
#
# ============================================================

DATA_PATH = "data/real/E2025-08-27_2026-08-27.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

# ------------------------------------------------------------
# 1. HANDLE ERROR CODE
# ------------------------------------------------------------

df.replace(3276.7, np.nan, inplace=True)


# ------------------------------------------------------------
# 2. CREATE TIMESTAMP
# ------------------------------------------------------------

print("\nCreating combined timestamp...")

df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

invalid = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid}")

df = df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------
# 3. SYSTEM 455 COLUMNS
# ------------------------------------------------------------

COLLECTOR = "1/Ana.1 - 1: 455_collector_temp"
BUFFER_BOTTOM = "1/Ana.2 - 2: 455_temp_buffer_bottom"
BUFFER_CENTRE = "1/Ana.3 - 3: 455_temp_buffer_centre"
BUFFER_TOP = "1/Ana.4 - 4: 455_temp_buffer_top"
SOLAR_RADIATION = "1/Ana.5 - 5: 455_solar_radiation"

SOLAR_FLOW = "1/Ana.9 - 2: 455_HM_solar_flow"
SOLAR_POWER = "1/Ana.10 - 3: 455_HM_solar_power"

HOTWATER_FLOW = "1/Ana.16 - 10: 455_HM_hotwater_flow"
HOTWATER_POWER = "1/Ana.17 - 11: 455_HM_hotwater_power"

SOLAR_PUMP = "1/Dig.1 - 1: 455_solar_pump"
ELECTRIC_HEATER = "1/Dig.3 - 3: 455_e-heater"


required_columns = [
    COLLECTOR,
    BUFFER_BOTTOM,
    BUFFER_CENTRE,
    BUFFER_TOP,
    SOLAR_RADIATION,
    SOLAR_FLOW,
    SOLAR_POWER,
    HOTWATER_FLOW,
    HOTWATER_POWER,
    SOLAR_PUMP,
    ELECTRIC_HEATER,
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column not found: {column}")


# Convert relevant columns to numeric
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ------------------------------------------------------------
# 4. HEADER
# ------------------------------------------------------------

print("\n")
print("=" * 90)
print("             STMaCS SYSTEM 455 ANOMALY INVESTIGATION")
print("=" * 90)

print("""
IMPORTANT:
This script DOES NOT delete, replace, interpolate, or modify
any suspicious measurements.

It only identifies potentially problematic observations and
examines their surrounding operating conditions.
""")


# ============================================================
# 5. BUFFER BOTTOM < 0°C
# ============================================================

print("\n")
print("=" * 90)
print("              INVESTIGATION 1: BUFFER BOTTOM < 0°C")
print("=" * 90)

buffer_bottom_bad = df[BUFFER_BOTTOM] < 0

bad_count = buffer_bottom_bad.sum()

print(f"\nNumber of observations below 0°C: {bad_count:,}")

if bad_count == 0:

    print("No observations below 0°C were found.")

else:

    bad_indices = df.index[buffer_bottom_bad]

    print("\nFirst suspicious observations:")
    print("-" * 90)

    print(
        f"{'Timestamp':<22}"
        f"{'Bottom':>10}"
        f"{'Centre':>10}"
        f"{'Top':>10}"
        f"{'Collector':>12}"
        f"{'Solar Rad':>12}"
        f"{'Solar Flow':>12}"
        f"{'Pump':>8}"
    )

    print("-" * 90)

    for idx in bad_indices[:20]:

        row = df.loc[idx]

        print(
            f"{str(row['timestamp']):<22}"
            f"{row[BUFFER_BOTTOM]:>10.2f}"
            f"{row[BUFFER_CENTRE]:>10.2f}"
            f"{row[BUFFER_TOP]:>10.2f}"
            f"{row[COLLECTOR]:>12.2f}"
            f"{row[SOLAR_RADIATION]:>12.2f}"
            f"{row[SOLAR_FLOW]:>12.2f}"
            f"{row[SOLAR_PUMP]:>8.0f}"
        )


# ============================================================
# 6. IDENTIFY CONSECUTIVE ANOMALY RUNS
# ============================================================

print("\n")
print("=" * 90)
print("              INVESTIGATION 2: ANOMALY RUNS")
print("=" * 90)

if bad_count > 0:

    bad_positions = np.where(buffer_bottom_bad.values)[0]

    runs = []

    start = bad_positions[0]
    previous = bad_positions[0]

    for position in bad_positions[1:]:

        if position == previous + 1:
            previous = position

        else:
            runs.append((start, previous))
            start = position
            previous = position

    runs.append((start, previous))

    print(f"\nNumber of separate anomaly runs: {len(runs)}")

    print("\n")
    print(
        f"{'Run':<8}"
        f"{'Records':>10}"
        f"{'Start':<22}"
        f"{'End':<22}"
        f"{'Duration':>12}"
    )

    print("-" * 78)

    for i, (start, end) in enumerate(runs, start=1):

        start_time = df.loc[start, "timestamp"]
        end_time = df.loc[end, "timestamp"]

        records = end - start + 1

        duration_minutes = (
            end_time - start_time
        ).total_seconds() / 60

        print(
            f"{i:<8}"
            f"{records:>10}"
            f"{str(start_time):<22}"
            f"{str(end_time):<22}"
            f"{duration_minutes:>10.1f} min"
        )


# ============================================================
# 7. SURROUNDING VALUES
# ============================================================

print("\n")
print("=" * 90)
print("           INVESTIGATION 3: SURROUNDING VALUES")
print("=" * 90)

if bad_count > 0:

    print("""
For each suspicious buffer-bottom measurement, the script
examines the measurements immediately before and after it.

This helps determine whether the value is:

    A. An isolated sensor spike
    B. Part of a sustained unusual condition
    C. A physically plausible event
    D. A likely sensor/data fault
""")

    # Show first 10 suspicious observations with neighbors
    shown = 0

    for idx in bad_indices:

        if shown >= 10:
            break

        print("\n")
        print("-" * 90)
        print(f"Suspicious record at index {idx}")
        print("-" * 90)

        start_idx = max(0, idx - 2)
        end_idx = min(len(df) - 1, idx + 2)

        print(
            f"{'Timestamp':<22}"
            f"{'Bottom':>10}"
            f"{'Centre':>10}"
            f"{'Top':>10}"
            f"{'Collector':>12}"
            f"{'Solar Rad':>12}"
            f"{'Solar Flow':>12}"
            f"{'Hot Flow':>12}"
        )

        print("-" * 90)

        for j in range(start_idx, end_idx + 1):

            row = df.loc[j]

            marker = " <-- FLAGGED" if j == idx else ""

            print(
                f"{str(row['timestamp']):<22}"
                f"{row[BUFFER_BOTTOM]:>10.2f}"
                f"{row[BUFFER_CENTRE]:>10.2f}"
                f"{row[BUFFER_TOP]:>10.2f}"
                f"{row[COLLECTOR]:>12.2f}"
                f"{row[SOLAR_RADIATION]:>12.2f}"
                f"{row[SOLAR_FLOW]:>12.2f}"
                f"{row[HOTWATER_FLOW]:>12.2f}"
                f"{marker}"
            )

        shown += 1


# ============================================================
# 8. TEMPERATURE RATE-OF-CHANGE ANALYSIS
# ============================================================

print("\n")
print("=" * 90)
print("          INVESTIGATION 4: TEMPERATURE RATE OF CHANGE")
print("=" * 90)

thermal_columns = {
    "Collector": COLLECTOR,
    "Buffer Bottom": BUFFER_BOTTOM,
    "Buffer Centre": BUFFER_CENTRE,
    "Buffer Top": BUFFER_TOP,
}

print("""
Calculating temperature change between consecutive observations.

Normal sampling is approximately 5 minutes, so large changes
between consecutive records may indicate either:

    - a genuine thermal event
    - a sensor spike
    - a data acquisition problem
""")

print("\n")
print(
    f"{'Sensor':<20}"
    f"{'Median |ΔT|':>15}"
    f"{'P99 |ΔT|':>15}"
    f"{'Max |ΔT|':>15}"
    f"{'Large Changes':>18}"
)

print("-" * 85)

large_change_indices = {}

for name, column in thermal_columns.items():

    delta = df[column].diff()

    abs_delta = delta.abs()

    median_change = abs_delta.median()
    p99_change = abs_delta.quantile(0.99)
    max_change = abs_delta.max()

    # Flag changes greater than 5°C between consecutive records
    large_changes = abs_delta > 5

    count_large = large_changes.sum()

    large_change_indices[name] = df.index[large_changes]

    print(
        f"{name:<20}"
        f"{median_change:>15.3f}"
        f"{p99_change:>15.3f}"
        f"{max_change:>15.3f}"
        f"{count_large:>18,}"
    )


# ============================================================
# 9. LARGE BUFFER-TOP CHANGES
# ============================================================

print("\n")
print("=" * 90)
print("          INVESTIGATION 5: LARGE BUFFER-TOP CHANGES")
print("=" * 90)

buffer_top_delta = df[BUFFER_TOP].diff()
large_top_change = buffer_top_delta.abs() > 5

count_large_top = large_top_change.sum()

print(f"\nBuffer-top changes greater than 5°C: {count_large_top:,}")

if count_large_top > 0:

    print("\nFirst 20 large changes:")
    print("-" * 90)

    print(
        f"{'Timestamp':<22}"
        f"{'Previous':>12}"
        f"{'Current':>12}"
        f"{'ΔT':>12}"
        f"{'Collector':>12}"
        f"{'Solar Rad':>12}"
        f"{'Hot Flow':>12}"
        f"{'Pump':>8}"
        f"{'Heater':>8}"
    )

    print("-" * 90)

    large_indices = df.index[large_top_change]

    for idx in large_indices[:20]:

        if idx == 0:
            continue

        row = df.loc[idx]
        previous = df.loc[idx - 1, BUFFER_TOP]
        delta = row[BUFFER_TOP] - previous

        print(
            f"{str(row['timestamp']):<22}"
            f"{previous:>12.2f}"
            f"{row[BUFFER_TOP]:>12.2f}"
            f"{delta:>12.2f}"
            f"{row[COLLECTOR]:>12.2f}"
            f"{row[SOLAR_RADIATION]:>12.2f}"
            f"{row[HOTWATER_FLOW]:>12.2f}"
            f"{row[SOLAR_PUMP]:>8.0f}"
            f"{row[ELECTRIC_HEATER]:>8.0f}"
        )


# ============================================================
# 10. THERMAL CONSISTENCY CHECK
# ============================================================

print("\n")
print("=" * 90)
print("              INVESTIGATION 6: THERMAL CONSISTENCY")
print("=" * 90)

print("""
For a stratified hot-water storage tank we generally expect:

    Buffer Top >= Buffer Centre >= Buffer Bottom

This is not an absolute rule under every transient operating
condition, but repeated violations can indicate sensor issues.
""")

top = df[BUFFER_TOP]
centre = df[BUFFER_CENTRE]
bottom = df[BUFFER_BOTTOM]

violations_top_centre = top < centre
violations_centre_bottom = centre < bottom
violations_top_bottom = top < bottom

print(
    f"\nTop < Centre violations:     "
    f"{violations_top_centre.sum():,}"
)

print(
    f"Centre < Bottom violations:  "
    f"{violations_centre_bottom.sum():,}"
)

print(
    f"Top < Bottom violations:     "
    f"{violations_top_bottom.sum():,}"
)


# ============================================================
# 11. COLLECTOR VS STORAGE TEMPERATURE
# ============================================================

print("\n")
print("=" * 90)
print("           INVESTIGATION 7: COLLECTOR / STORAGE")
print("=" * 90)

collector_minus_top = df[COLLECTOR] - df[BUFFER_TOP]

print(
    f"\nCollector - Buffer Top temperature difference:"
)

print(
    f"  Minimum:  {collector_minus_top.min():.2f}°C"
)

print(
    f"  Median:   {collector_minus_top.median():.2f}°C"
)

print(
    f"  P99:      {collector_minus_top.quantile(0.99):.2f}°C"
)

print(
    f"  Maximum:  {collector_minus_top.max():.2f}°C"
)

print(
    f"  Collector colder than buffer top: "
    f"{(collector_minus_top < 0).sum():,} records"
)


# ============================================================
# 12. FINAL FLAG SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("                    FINAL FLAG SUMMARY")
print("=" * 90)

print(
    f"""
Buffer-bottom < 0°C:
    {bad_count:,} observations

Buffer-top > 100°C:
    {(top > 100).sum():,} observations

Buffer-centre > 100°C:
    {(centre > 100).sum():,} observations

Collector > 180°C:
    {(df[COLLECTOR] > 180).sum():,} observations

Negative solar radiation:
    {(df[SOLAR_RADIATION] < 0).sum():,} observations

Negative solar flow:
    {(df[SOLAR_FLOW] < 0).sum():,} observations

Negative hot-water flow:
    {(df[HOTWATER_FLOW] < 0).sum():,} observations

Top < Centre:
    {violations_top_centre.sum():,} observations

Centre < Bottom:
    {violations_centre_bottom.sum():,} observations

Top < Bottom:
    {violations_top_bottom.sum():,} observations
"""
)

print("""
============================================================

NO DATA HAS BEEN DELETED OR MODIFIED.

The purpose of this investigation is to understand the data
before deciding on cleaning rules.

NEXT DECISION:
    We will use these results to define scientifically
    defensible preprocessing rules for the ML dataset.

============================================================
""")

print("ANOMALY INVESTIGATION COMPLETE.")