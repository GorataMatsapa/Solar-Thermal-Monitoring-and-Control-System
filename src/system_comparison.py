import pandas as pd
import numpy as np

# ============================================================
# STMaCS SYSTEM COMPARISON
# ============================================================

DATA_PATH = "data/real/E2025-08-27_2026-08-27.csv"

SYSTEM_IDS = ["455", "457", "459"]

# Electrical/error code used by the monitoring system
ERROR_CODE = 3276.7

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

# Replace known electrical error code with NaN
df.replace(ERROR_CODE, np.nan, inplace=True)


# ============================================================
# 2. FIND TIMESTAMP COLUMN
# ============================================================

print("\nSearching for timestamp column...")

timestamp_candidates = [
    col for col in df.columns
    if any(keyword in col.lower()
           for keyword in ["timestamp", "datetime", "date_time", "time", "date"])
]

print("Possible timestamp columns:")
for col in timestamp_candidates:
    print(f"  - {col}")

# Try to automatically identify the best timestamp column
timestamp_col = None

preferred_keywords = [
    "timestamp",
    "datetime",
    "date_time"
]

for keyword in preferred_keywords:
    matches = [c for c in df.columns if keyword in c.lower()]
    if matches:
        timestamp_col = matches[0]
        break

# If no preferred column was found, use the first candidate
if timestamp_col is None and timestamp_candidates:
    timestamp_col = timestamp_candidates[0]

if timestamp_col is None:
    raise ValueError(
        "Could not automatically identify a timestamp column. "
        "Please inspect the printed column names and set timestamp_col manually."
    )

print(f"\nUsing timestamp column: {timestamp_col}")

# Convert timestamp
df[timestamp_col] = pd.to_datetime(
    df[timestamp_col],
    errors="coerce"
)

valid_timestamps = df[timestamp_col].dropna()

if len(valid_timestamps) == 0:
    raise ValueError("Timestamp column could not be converted to valid datetime values.")


# ============================================================
# 3. MAP SYSTEM COLUMNS
# ============================================================

systems = {}

for sys_id in SYSTEM_IDS:
    systems[sys_id] = [
        col for col in df.columns
        if sys_id in str(col)
    ]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def completeness(series):
    """Return percentage of non-null values."""
    return series.notna().mean() * 100


def percentile_summary(series):
    """
    Return robust statistics for a numeric series.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if len(numeric) == 0:
        return None

    return {
        "min": numeric.min(),
        "p1": numeric.quantile(0.01),
        "p5": numeric.quantile(0.05),
        "median": numeric.median(),
        "p95": numeric.quantile(0.95),
        "p99": numeric.quantile(0.99),
        "max": numeric.max()
    }


def format_interval(seconds):
    """
    Convert seconds into a human-readable interval.
    """
    if pd.isna(seconds):
        return "N/A"

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f} seconds"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.2f} minutes"

    hours = minutes / 60

    if hours < 24:
        return f"{hours:.2f} hours"

    days = hours / 24

    return f"{days:.2f} days"


def analyze_logging_intervals(system_df, sys_id):
    """
    Analyse timestamp intervals for a system.

    Since the systems share the main timestamp column, this checks:
    1. Global timestamp interval
    2. Whether rows containing valid system data follow the same interval
    """

    timestamps = system_df[timestamp_col].dropna().sort_values()

    if len(timestamps) < 2:
        return None

    # Difference between consecutive timestamps
    deltas = timestamps.diff().dropna()

    # Convert to seconds
    delta_seconds = deltas.dt.total_seconds()

    # Remove zero/negative intervals
    delta_seconds = delta_seconds[delta_seconds > 0]

    if len(delta_seconds) == 0:
        return None

    # Most common interval
    interval_counts = delta_seconds.value_counts()

    dominant_interval = interval_counts.index[0]
    dominant_count = interval_counts.iloc[0]

    # Median interval
    median_interval = delta_seconds.median()

    # Gap threshold
    # Anything greater than 2x the dominant interval is considered
    # a potential logging gap.
    gap_threshold = dominant_interval * 2

    gaps = delta_seconds[delta_seconds > gap_threshold]

    return {
        "records": len(timestamps),
        "start": timestamps.min(),
        "end": timestamps.max(),
        "duration_days": (
            timestamps.max() - timestamps.min()
        ).total_seconds() / 86400,

        "dominant_interval_seconds": dominant_interval,
        "dominant_interval_count": dominant_count,

        "dominant_interval_percentage": (
            dominant_count / len(delta_seconds) * 100
        ),

        "median_interval_seconds": median_interval,
        "minimum_interval_seconds": delta_seconds.min(),
        "maximum_interval_seconds": delta_seconds.max(),

        "unique_intervals": delta_seconds.nunique(),

        "number_of_gaps": len(gaps),

        "largest_gap_seconds": (
            gaps.max() if len(gaps) > 0 else 0
        )
    }


# ============================================================
# 5. HEADER
# ============================================================

print("\n")
print("=" * 72)
print("          STMaCS SYSTEM COMPARISON FEASIBILITY REPORT")
print("=" * 72)


# ============================================================
# 6. GLOBAL TIMESTAMP INFORMATION
# ============================================================

print("\n### GLOBAL DATASET TIME COVERAGE")

print(f"Timestamp column:       {timestamp_col}")
print(f"First timestamp:        {valid_timestamps.min()}")
print(f"Last timestamp:         {valid_timestamps.max()}")

global_deltas = (
    valid_timestamps
    .sort_values()
    .diff()
    .dropna()
    .dt.total_seconds()
)

global_deltas = global_deltas[global_deltas > 0]

if len(global_deltas) > 0:

    global_counts = global_deltas.value_counts()

    dominant_global_interval = global_counts.index[0]

    print(
        f"Dominant logging interval: "
        f"{format_interval(dominant_global_interval)}"
    )

    print(
        f"Median logging interval:   "
        f"{format_interval(global_deltas.median())}"
    )

    print(
        f"Minimum interval:          "
        f"{format_interval(global_deltas.min())}"
    )

    print(
        f"Maximum interval:          "
        f"{format_interval(global_deltas.max())}"
    )

    print(
        f"Unique intervals observed: "
        f"{global_deltas.nunique()}"
    )

    print(
        f"Dominant interval occurs:  "
        f"{global_counts.iloc[0] / len(global_deltas) * 100:.2f}% "
        f"of the time"
    )


# ============================================================
# 7. SYSTEM-BY-SYSTEM ANALYSIS
# ============================================================

system_results = {}

for sys_id, cols in systems.items():

    print("\n")
    print("=" * 72)
    print(f"                        SYSTEM {sys_id}")
    print("=" * 72)

    if not cols:
        print("NO COLUMNS FOUND.")
        continue

    print(f"\nTotal mapped columns: {len(cols)}")

    # --------------------------------------------------------
    # Overall completeness
    # --------------------------------------------------------

    total_cells = len(cols) * len(df)

    non_null_cells = (
        df[cols]
        .notna()
        .sum()
        .sum()
    )

    completion_rate = (
        non_null_cells / total_cells * 100
    )

    print(
        f"Overall data completion: "
        f"{completion_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Dead columns
    # --------------------------------------------------------

    dead_cols = [
        col for col in cols
        if df[col].isna().all()
    ]

    print(
        f"Completely empty/dead columns: "
        f"{len(dead_cols)}"
    )

    if dead_cols:
        print("Dead columns:")
        for col in dead_cols:
            print(f"  - {col}")

    # --------------------------------------------------------
    # TIME COVERAGE
    # --------------------------------------------------------

    # Rows where at least one system-specific variable exists
    system_valid_mask = df[cols].notna().any(axis=1)

    system_df = df.loc[
        system_valid_mask,
        [timestamp_col] + cols
    ].copy()

    system_df = system_df.dropna(
        subset=[timestamp_col]
    )

    print("\n### TIME COVERAGE")

    if len(system_df) > 0:

        start_time = system_df[timestamp_col].min()
        end_time = system_df[timestamp_col].max()

        duration = (
            end_time - start_time
        ).total_seconds() / 86400

        print(f"First valid system record: {start_time}")
        print(f"Last valid system record:  {end_time}")
        print(f"Coverage duration:          {duration:.2f} days")
        print(f"Rows containing data:       {len(system_df):,}")

    else:
        print("No valid timestamps for this system.")

    # --------------------------------------------------------
    # LOGGING INTERVAL
    # --------------------------------------------------------

    print("\n### LOGGING INTERVAL")

    interval_info = analyze_logging_intervals(
        system_df,
        sys_id
    )

    if interval_info:

        print(
            "Dominant logging interval: "
            f"{format_interval(interval_info['dominant_interval_seconds'])}"
        )

        print(
            "Median logging interval:   "
            f"{format_interval(interval_info['median_interval_seconds'])}"
        )

        print(
            "Minimum interval:          "
            f"{format_interval(interval_info['minimum_interval_seconds'])}"
        )

        print(
            "Maximum interval:          "
            f"{format_interval(interval_info['maximum_interval_seconds'])}"
        )

        print(
            "Unique intervals observed: "
            f"{interval_info['unique_intervals']}"
        )

        print(
            "Dominant interval coverage:"
            f" {interval_info['dominant_interval_percentage']:.2f}%"
        )

        print(
            "Potential logging gaps:     "
            f"{interval_info['number_of_gaps']}"
        )

        if interval_info["largest_gap_seconds"] > 0:
            print(
                "Largest detected gap:      "
                f"{format_interval(interval_info['largest_gap_seconds'])}"
            )

    else:
        print("Unable to determine logging interval.")

    # --------------------------------------------------------
    # SENSOR INVENTORY
    # --------------------------------------------------------

    print("\n### SENSOR / COLUMN INVENTORY")

    print(
        f"{'Column':55} "
        f"{'Complete':>10} "
        f"{'Valid':>10} "
        f"{'Missing':>10}"
    )

    print("-" * 90)

    for col in cols:

        series = df[col]

        valid = series.notna().sum()
        missing = series.isna().sum()
        comp = completeness(series)

        print(
            f"{str(col)[:55]:55} "
            f"{comp:9.2f}% "
            f"{valid:10,} "
            f"{missing:10,}"
        )

    # --------------------------------------------------------
    # NUMERIC SENSOR STATISTICS
    # --------------------------------------------------------

    print("\n### NUMERIC SENSOR STATISTICS")

    numeric_cols = []

    for col in cols:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    print(
        f"{'Column':45} "
        f"{'Min':>10} "
        f"{'P1':>10} "
        f"{'Median':>10} "
        f"{'P99':>10} "
        f"{'Max':>10}"
    )

    print("-" * 100)

    for col in numeric_cols:

        stats = percentile_summary(df[col])

        if stats is None:
            continue

        print(
            f"{str(col)[:45]:45} "
            f"{stats['min']:10.2f} "
            f"{stats['p1']:10.2f} "
            f"{stats['median']:10.2f} "
            f"{stats['p99']:10.2f} "
            f"{stats['max']:10.2f}"
        )

    # Save results for later comparison
    system_results[sys_id] = {
        "columns": len(cols),
        "completion": completion_rate,
        "dead_columns": len(dead_cols),
        "interval": interval_info
    }


# ============================================================
# 8. CROSS-SYSTEM SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("                     CROSS-SYSTEM SUMMARY")
print("=" * 72)

print(
    f"\n{'System':<10}"
    f"{'Columns':>10}"
    f"{'Complete':>14}"
    f"{'Dead':>10}"
    f"{'Logging Interval':>22}"
)

print("-" * 72)

for sys_id in SYSTEM_IDS:

    if sys_id not in system_results:
        continue

    result = system_results[sys_id]

    interval = result["interval"]

    if interval:
        interval_text = format_interval(
            interval["dominant_interval_seconds"]
        )
    else:
        interval_text = "N/A"

    print(
        f"{sys_id:<10}"
        f"{result['columns']:>10}"
        f"{result['completion']:>13.2f}%"
        f"{result['dead_columns']:>10}"
        f"{interval_text:>22}"
    )


# ============================================================
# 9. IMPORTANT VARIABLES CROSS-CHECK
# ============================================================

print("\n")
print("=" * 72)
print("              IMPORTANT VARIABLE AVAILABILITY")
print("=" * 72)

keywords = {
    "Collector temperature": [
        "collector",
        "t.collector"
    ],

    "Buffer top": [
        "buffer_top"
    ],

    "Buffer middle": [
        "buffer_middle",
        "buffer_mid"
    ],

    "Buffer bottom": [
        "buffer_bottom",
        "buffer_bot"
    ],

    "Ambient temperature": [
        "ambient",
        "t.ambient"
    ],

    "Solar radiation": [
        "solar_radiation",
        "irradiance",
        "radiation"
    ],

    "Hot water flow": [
        "hot_water",
        "hotwater",
        "dhw",
        "flow"
    ],

    "Solar flow": [
        "solar_flow"
    ],

    "Pump": [
        "pump"
    ],

    "System state": [
        "state",
        "status"
    ]
}


for variable, search_terms in keywords.items():

    print(f"\n{variable}:")

    for sys_id, cols in systems.items():

        matches = []

        for col in cols:

            col_lower = col.lower()

            if any(
                term.lower() in col_lower
                for term in search_terms
            ):
                matches.append(col)

        if matches:
            print(f"  {sys_id}: FOUND")
            for match in matches:
                print(f"       - {match}")
        else:
            print(f"  {sys_id}: NOT FOUND")


# ============================================================
# 10. FINAL PRELIMINARY ASSESSMENT
# ============================================================

print("\n")
print("=" * 72)
print("                 PRELIMINARY ASSESSMENT")
print("=" * 72)

print("""
This report is a DATA SCREENING tool.

Do NOT select a final modeling system based only on
completion percentage.

The final decision should consider:

1. Predictor availability
2. Data completeness
3. Logging consistency
4. Time coverage
5. Sensor plausibility
6. Thermal operating range
7. Availability of the target variable
8. Ability to construct lagged/time-series features
9. Whether systems 455, 457 and 459 are physically comparable

The next stage should be feature/target feasibility analysis
before training the machine-learning model.
""")

print("=" * 72)
print("                    ANALYSIS COMPLETE")
print("=" * 72)