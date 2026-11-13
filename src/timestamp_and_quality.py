import pandas as pd
import numpy as np

# ============================================================
# STMaCS — TIMESTAMP & DATA QUALITY AUDIT
# ============================================================

DATA_PATH = "data/real/E2025-08-27_2026-08-27.csv"

SYSTEM_IDS = ["455", "457", "459"]

ERROR_CODE = 3276.7

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Rows loaded:    {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

# Replace known electrical error code
df.replace(ERROR_CODE, np.nan, inplace=True)


# ============================================================
# 2. CREATE TRUE TIMESTAMP
# ============================================================

print("\nCreating combined timestamp from Date + Time...")

if "Date" not in df.columns:
    raise ValueError("Column 'Date' was not found.")

if "Time" not in df.columns:
    raise ValueError("Column 'Time' was not found.")

df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps:,}")

if invalid_timestamps == len(df):
    raise ValueError(
        "No valid timestamps could be created. "
        "Check the Date and Time columns."
    )

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 3. GLOBAL TIMESTAMP ANALYSIS
# ============================================================

print("\n")
print("=" * 72)
print("                  GLOBAL TIMESTAMP ANALYSIS")
print("=" * 72)

valid_time = df["timestamp"].dropna()

print(f"\nFirst timestamp: {valid_time.min()}")
print(f"Last timestamp:  {valid_time.max()}")

duration_days = (
    valid_time.max() - valid_time.min()
).total_seconds() / 86400

print(f"Total time span: {duration_days:.2f} days")

# Calculate intervals
global_deltas = (
    valid_time
    .diff()
    .dropna()
    .dt.total_seconds()
)

global_deltas = global_deltas[global_deltas > 0]

if len(global_deltas) == 0:
    raise ValueError("Could not calculate timestamp intervals.")

interval_counts = global_deltas.value_counts()

dominant_interval = interval_counts.index[0]

print("\n### GLOBAL LOGGING INTERVAL")

print(
    f"Dominant interval:       "
    f"{dominant_interval:.2f} seconds"
)

print(
    f"Dominant interval:       "
    f"{dominant_interval / 60:.2f} minutes"
)

print(
    f"Median interval:         "
    f"{global_deltas.median():.2f} seconds"
)

print(
    f"Minimum interval:        "
    f"{global_deltas.min():.2f} seconds"
)

print(
    f"Maximum interval:        "
    f"{global_deltas.max():.2f} seconds"
)

print(
    f"Unique intervals:        "
    f"{global_deltas.nunique()}"
)

print(
    f"Dominant interval share: "
    f"{interval_counts.iloc[0] / len(global_deltas) * 100:.2f}%"
)


# ============================================================
# 4. SHOW MOST COMMON INTERVALS
# ============================================================

print("\n### TOP 10 MOST COMMON LOGGING INTERVALS")

print(
    f"{'Interval (seconds)':>20}"
    f"{'Interval (minutes)':>22}"
    f"{'Occurrences':>15}"
    f"{'Percentage':>15}"
)

print("-" * 75)

for seconds, count in interval_counts.head(10).items():

    percentage = count / len(global_deltas) * 100

    print(
        f"{seconds:>20.2f}"
        f"{seconds / 60:>22.2f}"
        f"{count:>15,}"
        f"{percentage:>14.2f}%"
    )


# ============================================================
# 5. GLOBAL TIMESTAMP GAPS
# ============================================================

print("\n### GLOBAL LOGGING GAPS")

# Anything greater than twice the dominant interval
gap_threshold = dominant_interval * 2

gaps = global_deltas[
    global_deltas > gap_threshold
]

print(
    f"Gap threshold: "
    f"{gap_threshold / 60:.2f} minutes"
)

print(
    f"Number of potential gaps: "
    f"{len(gaps):,}"
)

if len(gaps) > 0:

    print(
        f"Largest gap: "
        f"{gaps.max() / 3600:.2f} hours"
    )

    print("\nLargest gaps:")

    gap_indices = global_deltas[
        global_deltas > gap_threshold
    ].nlargest(10).index

    for idx in gap_indices:

        previous_time = df.loc[idx - 1, "timestamp"]
        current_time = df.loc[idx, "timestamp"]

        gap = (
            current_time - previous_time
        ).total_seconds()

        print(
            f"  {previous_time} → {current_time} "
            f"({gap / 3600:.2f} hours)"
        )

else:

    print("No major global timestamp gaps detected.")


# ============================================================
# 6. SYSTEM COLUMN MAPPING
# ============================================================

systems = {}

for sys_id in SYSTEM_IDS:

    systems[sys_id] = [
        col for col in df.columns
        if sys_id in str(col)
    ]


# ============================================================
# 7. SYSTEM-SPECIFIC TIMESTAMP ANALYSIS
# ============================================================

print("\n")
print("=" * 72)
print("               SYSTEM-SPECIFIC TIMESTAMP ANALYSIS")
print("=" * 72)

system_interval_results = {}

for sys_id, cols in systems.items():

    print(f"\n### SYSTEM {sys_id}")

    if not cols:

        print("No columns found.")
        continue

    # Identify rows where at least one system measurement exists
    valid_system_rows = df[cols].notna().any(axis=1)

    system_times = (
        df.loc[
            valid_system_rows,
            "timestamp"
        ]
        .dropna()
        .sort_values()
    )

    if len(system_times) < 2:

        print("Not enough timestamps for interval analysis.")
        continue

    deltas = (
        system_times
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    deltas = deltas[deltas > 0]

    counts = deltas.value_counts()

    dominant = counts.index[0]

    system_interval_results[sys_id] = dominant

    print(f"Valid timestamped records: {len(system_times):,}")

    print(
        f"First record:             "
        f"{system_times.min()}"
    )

    print(
        f"Last record:              "
        f"{system_times.max()}"
    )

    print(
        f"Dominant logging interval:"
        f" {dominant / 60:.2f} minutes"
    )

    print(
        f"Median logging interval:  "
        f"{deltas.median() / 60:.2f} minutes"
    )

    print(
        f"Minimum interval:         "
        f"{deltas.min():.2f} seconds"
    )

    print(
        f"Maximum interval:         "
        f"{deltas.max() / 3600:.2f} hours"
    )

    print(
        f"Dominant interval share:  "
        f"{counts.iloc[0] / len(deltas) * 100:.2f}%"
    )

    system_gaps = deltas[
        deltas > dominant * 2
    ]

    print(
        f"Potential gaps:            "
        f"{len(system_gaps):,}"
    )

    if len(system_gaps) > 0:

        print(
            f"Largest gap:              "
            f"{system_gaps.max() / 3600:.2f} hours"
        )


# ============================================================
# 8. THERMAL SENSOR QUALITY
# ============================================================

print("\n")
print("=" * 72)
print("                  THERMAL SENSOR QUALITY")
print("=" * 72)


# Explicitly map important variables.
# This avoids the broad keyword matching problem in the previous script.

thermal_columns = {

    "455 Collector": "1/Ana.1 - 1: 455_collector_temp",
    "455 Buffer Bottom": "1/Ana.2 - 2: 455_temp_buffer_bottom",
    "455 Buffer Centre": "1/Ana.3 - 3: 455_temp_buffer_centre",
    "455 Buffer Top": "1/Ana.4 - 4: 455_temp_buffer_top",

    "457 Collector": "2/Ana.1 - 1: 457_T.collector",
    "457 Buffer Bottom": "2/Ana.2 - 2: 457_T.buffer_bottom",
    "457 Buffer Centre": "2/Ana.3 - 3: 457_T.buffer_centre",
    "457 Buffer Top": "2/Ana.4 - 4: 457_T.buffer_top",

    "459 Collector": "3/Ana.1 - 1: 459_T.collector",
    "459 Buffer Bottom": "3/Ana.2 - 2: 459_T.buffer_bottom",
    "459 Buffer Centre": "3/Ana.3 - 3: 459_T.buffer_centre",
    "459 Buffer Top": "3/Ana.4 - 4: 459_T.buffer_top",
}


print(
    f"\n{'Sensor':<25}"
    f"{'Valid':>10}"
    f"{'Missing':>10}"
    f"{'Min':>10}"
    f"{'P1':>10}"
    f"{'Median':>10}"
    f"{'P99':>10}"
    f"{'Max':>10}"
)

print("-" * 95)

for name, column in thermal_columns.items():

    if column not in df.columns:

        print(
            f"{name:<25}"
            f"{'COLUMN NOT FOUND':>10}"
        )

        continue

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    valid = series.notna().sum()
    missing = series.isna().sum()

    if valid == 0:

        print(
            f"{name:<25}"
            f"{valid:>10,}"
            f"{missing:>10,}"
            f"{'NO DATA':>10}"
        )

        continue

    print(
        f"{name:<25}"
        f"{valid:>10,}"
        f"{missing:>10,}"
        f"{series.min():>10.2f}"
        f"{series.quantile(0.01):>10.2f}"
        f"{series.median():>10.2f}"
        f"{series.quantile(0.99):>10.2f}"
        f"{series.max():>10.2f}"
    )


# ============================================================
# 9. EXTREME VALUE INVESTIGATION
# ============================================================

print("\n")
print("=" * 72)
print("                 EXTREME VALUE INVESTIGATION")
print("=" * 72)

# These are NOT automatically deleted.
# We only identify them so we can decide later whether
# they are genuine operating conditions or sensor errors.

extreme_limits = {

    "455 Collector": (
        "1/Ana.1 - 1: 455_collector_temp",
        -20,
        180
    ),

    "455 Buffer Bottom": (
        "1/Ana.2 - 2: 455_temp_buffer_bottom",
        0,
        100
    ),

    "455 Buffer Centre": (
        "1/Ana.3 - 3: 455_temp_buffer_centre",
        0,
        100
    ),

    "455 Buffer Top": (
        "1/Ana.4 - 4: 455_temp_buffer_top",
        0,
        100
    ),

    "457 Collector": (
        "2/Ana.1 - 1: 457_T.collector",
        -20,
        180
    ),

    "457 Buffer Bottom": (
        "2/Ana.2 - 2: 457_T.buffer_bottom",
        0,
        100
    ),

    "457 Buffer Centre": (
        "2/Ana.3 - 3: 457_T.buffer_centre",
        0,
        100
    ),

    "457 Buffer Top": (
        "2/Ana.4 - 4: 457_T.buffer_top",
        0,
        100
    ),

    "459 Collector": (
        "3/Ana.1 - 1: 459_T.collector",
        -20,
        180
    ),

    "459 Buffer Bottom": (
        "3/Ana.2 - 2: 459_T.buffer_bottom",
        0,
        100
    ),

    "459 Buffer Centre": (
        "3/Ana.3 - 3: 459_T.buffer_centre",
        0,
        100
    ),

    "459 Buffer Top": (
        "3/Ana.4 - 4: 459_T.buffer_top",
        0,
        100
    ),
}


for name, (column, lower, upper) in extreme_limits.items():

    if column not in df.columns:
        continue

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    below = series < lower
    above = series > upper

    below_count = below.sum()
    above_count = above.sum()

    print(f"\n{name}")

    print(
        f"  Expected screening range: "
        f"{lower}°C to {upper}°C"
    )

    print(
        f"  Below lower limit: "
        f"{below_count:,}"
    )

    print(
        f"  Above upper limit: "
        f"{above_count:,}"
    )

    # Show examples of extreme values
    extreme_mask = below | above

    if extreme_mask.any():

        extreme_data = df.loc[
            extreme_mask,
            ["timestamp", column]
        ].copy()

        extreme_data = extreme_data.sort_values(
            "timestamp"
        )

        print("  First extreme observations:")

        for _, row in extreme_data.head(5).iterrows():

            print(
                f"    {row['timestamp']} → "
                f"{row[column]}°C"
            )


# ============================================================
# 10. MISSING-DATA RUNS FOR KEY SENSORS
# ============================================================

print("\n")
print("=" * 72)
print("                MISSING-DATA RUN ANALYSIS")
print("=" * 72)

key_columns = {

    "455 Collector":
        "1/Ana.1 - 1: 455_collector_temp",

    "455 Buffer Top":
        "1/Ana.4 - 4: 455_temp_buffer_top",

    "455 Solar Radiation":
        "1/Ana.5 - 5: 455_solar_radiation",

    "457 Collector":
        "2/Ana.1 - 1: 457_T.collector",

    "457 Buffer Top":
        "2/Ana.4 - 4: 457_T.buffer_top",

    "459 Collector":
        "3/Ana.1 - 1: 459_T.collector",

    "459 Buffer Top":
        "3/Ana.4 - 4: 459_T.buffer_top",
}


for name, column in key_columns.items():

    if column not in df.columns:
        continue

    missing_mask = df[column].isna()

    missing_count = missing_mask.sum()

    print(f"\n{name}")

    print(
        f"  Missing records: "
        f"{missing_count:,}"
    )

    if missing_count == 0:
        print("  No missing records.")
        continue

    # Identify consecutive missing runs
    groups = (
        missing_mask
        .ne(missing_mask.shift())
        .cumsum()
    )

    missing_runs = (
        df[missing_mask]
        .groupby(groups[missing_mask])
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            count=("timestamp", "count")
        )
    )

    missing_runs["duration_minutes"] = (
        missing_runs["end"] -
        missing_runs["start"]
    ).dt.total_seconds() / 60

    print(
        f"  Missing-data runs: "
        f"{len(missing_runs):,}"
    )

    print("  Largest missing runs:")

    largest_runs = missing_runs.sort_values(
        "count",
        ascending=False
    ).head(5)

    for _, row in largest_runs.iterrows():

        print(
            f"    {row['start']} → {row['end']} | "
            f"{int(row['count']):,} records | "
            f"{row['duration_minutes']:.1f} minutes"
        )


# ============================================================
# 11. FINAL CROSS-SYSTEM SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("                    FINAL SUMMARY")
print("=" * 72)

print(
    f"\n{'System':<10}"
    f"{'Records':>15}"
    f"{'Logging Interval':>22}"
)

print("-" * 50)

for sys_id in SYSTEM_IDS:

    cols = systems.get(sys_id, [])

    if not cols:
        continue

    valid_rows = df[cols].notna().any(axis=1).sum()

    interval = system_interval_results.get(
        sys_id,
        np.nan
    )

    if not pd.isna(interval):

        interval_text = (
            f"{interval / 60:.2f} min"
        )

    else:

        interval_text = "N/A"

    print(
        f"{sys_id:<10}"
        f"{valid_rows:>15,}"
        f"{interval_text:>22}"
    )


print("\n")
print("=" * 72)
print("                    AUDIT COMPLETE")
print("=" * 72)

print("""
IMPORTANT:

This script DOES NOT delete or modify suspicious measurements.

It only identifies:

- Actual Date + Time logging interval
- Logging gaps
- Missing-data runs
- Sensor completeness
- Extreme thermal values
- Potential sensor anomalies


""")

print("=" * 72)