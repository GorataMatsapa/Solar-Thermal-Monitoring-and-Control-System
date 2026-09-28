import pandas as pd
import numpy as np

# ============================================================
# SYSTEM 455 — TEMPORAL / LEAKAGE AUDIT
# ============================================================

FILE = "data/processed/system455_clean.csv"

print("=" * 70)
print("SYSTEM 455 — TEMPORAL / LEAKAGE AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"\nRows: {len(df):,}")
print(f"First timestamp: {df['timestamp'].iloc[0]}")
print(f"Last timestamp:  {df['timestamp'].iloc[-1]}")

# ------------------------------------------------------------
# 2. SAMPLING GAPS
# ------------------------------------------------------------

df["interval_minutes"] = (
    df["timestamp"].diff().dt.total_seconds() / 60
)

gaps = df[df["interval_minutes"] > 10].copy()

print("\n" + "=" * 70)
print("1. LOGGING GAPS > 10 MINUTES")
print("=" * 70)

print(f"\nNumber of gaps > 10 minutes: {len(gaps)}")

if len(gaps) > 0:
    print("\nGap details:")

    for idx, row in gaps.iterrows():
        previous_timestamp = df.loc[idx - 1, "timestamp"]

        print(
            f"  {previous_timestamp} → {row['timestamp']} "
            f"= {row['interval_minutes']:.2f} minutes"
        )

# ------------------------------------------------------------
# 3. CHECK HOW MANY ROW-BASED LAGS SPAN ABNORMAL GAPS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. ROW-BASED LAG TEMPORAL VALIDITY")
print("=" * 70)

lag_values = [1, 3, 6, 12]

for lag in lag_values:

    # Timestamp difference between current row and lagged row
    lag_time_minutes = (
        df["timestamp"] - df["timestamp"].shift(lag)
    ).dt.total_seconds() / 60

    expected_minutes = lag * 5

    # Define a tolerance around the expected time.
    # We allow 10% deviation plus 1 minute.
    tolerance = max(1.0, expected_minutes * 0.10)

    abnormal = lag_time_minutes[
        (lag_time_minutes - expected_minutes).abs() > tolerance
    ]

    print(f"\nlag_{lag}")
    print(f"  Expected time: {expected_minutes} minutes")
    print(f"  Tolerance: ±{tolerance:.2f} minutes")
    print(f"  Valid rows: {lag_time_minutes.notna().sum():,}")
    print(f"  Abnormal lag intervals: {len(abnormal):,}")

    if len(abnormal) > 0:
        print(
            f"  Percentage abnormal: "
            f"{len(abnormal) / lag_time_minutes.notna().sum() * 100:.4f}%"
        )

        print(
            f"  Minimum actual interval: "
            f"{abnormal.min():.2f} minutes"
        )

        print(
            f"  Maximum actual interval: "
            f"{abnormal.max():.2f} minutes"
        )

# ------------------------------------------------------------
# 4. IDENTIFY TRAIN / TEST BOUNDARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. TRAIN / TEST BOUNDARY")
print("=" * 70)

# IMPORTANT:
# This reproduces the 80/20 row-based split used in the dataset builder.

train_rows = int(len(df) * 0.80)

train_df = df.iloc[:train_rows].copy()
test_df = df.iloc[train_rows:].copy()

train_end = train_df["timestamp"].iloc[-1]
test_start = test_df["timestamp"].iloc[0]

print(f"\nTraining rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(f"\nLast training timestamp:")
print(f"  {train_end}")

print(f"\nFirst testing timestamp:")
print(f"  {test_start}")

print(
    f"\nTime between train end and test start: "
    f"{(test_start - train_end).total_seconds() / 60:.2f} minutes"
)

# ------------------------------------------------------------
# 5. TARGET BOUNDARY LEAKAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. TRAINING TARGETS CROSSING INTO TEST PERIOD")
print("=" * 70)

horizons = [5, 15, 30, 60]

for horizon in horizons:

    target_time = train_df["timestamp"] + pd.to_timedelta(
        horizon, unit="minutes"
    )

    crossing = train_df[target_time > test_start]

    print(f"\n{horizon}-minute horizon")
    print(f"  Training samples: {len(train_df):,}")
    print(
        f"  Targets extending into test period: "
        f"{len(crossing):,}"
    )

    if len(crossing) > 0:

        percentage = len(crossing) / len(train_df) * 100

        print(f"  Percentage of training set: {percentage:.4f}%")

        print(
            f"  First affected training timestamp: "
            f"{crossing['timestamp'].iloc[0]}"
        )

        print(
            f"  Last affected training timestamp: "
            f"{crossing['timestamp'].iloc[-1]}"
        )

# ------------------------------------------------------------
# 6. EXACT NUMBER OF SAFE TRAINING SAMPLES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. SAFE TRAINING SAMPLE COUNTS")
print("=" * 70)

for horizon in horizons:

    target_time = train_df["timestamp"] + pd.to_timedelta(
        horizon, unit="minutes"
    )

    safe = train_df[target_time <= test_start]

    removed = len(train_df) - len(safe)

    print(f"\n{horizon}-minute horizon")
    print(f"  Original training rows: {len(train_df):,}")
    print(f"  Safe training rows:     {len(safe):,}")
    print(f"  Rows to exclude:        {removed:,}")

# ------------------------------------------------------------
# 7. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print("""
This diagnostic does NOT modify any files.

It checks:

1. Major logging gaps.
2. Whether row-based lag features correspond to their intended
   physical time intervals.
3. The exact train/test temporal boundary.
4. Whether training targets extend into the test period.
5. How many training samples would need to be excluded to ensure
   every training target remains entirely inside the training period.
""")

print("=" * 70)
print("END OF AUDIT")
print("=" * 70)