import pandas as pd

file = "data/processed/system455_clean.csv"

df = pd.read_csv(file)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

intervals = df["timestamp"].diff().dropna().dt.total_seconds() / 60

print("\n=== SYSTEM 455 SAMPLING INTERVAL ===")
print(f"Rows: {len(df):,}")
print(f"First timestamp: {df['timestamp'].iloc[0]}")
print(f"Last timestamp:  {df['timestamp'].iloc[-1]}")

print("\nInterval statistics:")
print(f"Median:  {intervals.median():.2f} minutes")
print(f"Mean:    {intervals.mean():.2f} minutes")
print(f"Minimum: {intervals.min():.2f} minutes")
print(f"Maximum: {intervals.max():.2f} minutes")

print("\nMost common intervals:")
print(intervals.value_counts().head(15))

print("\nIntervals <= 10 minutes:")
print(intervals[intervals <= 10].value_counts().sort_index())

print("\nIntervals > 10 minutes:")
print(intervals[intervals > 10].describe())