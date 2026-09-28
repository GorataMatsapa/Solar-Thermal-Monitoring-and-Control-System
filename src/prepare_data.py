import pandas as pd

# -----------------------------
# 1. Load synthetic dataset
# -----------------------------

input_path = (
    "data/synthetic/"
    "synthetic_solar_thermal_data.csv"
)

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")
print(f"Rows: {len(df)}")


# -----------------------------
# 2. Define ML features
# -----------------------------

features = [
    "hour",
    "day_of_year",
    "ambient_temperature",
    "solar_irradiance",
    "flow_rate",
    "water_temperature",
    "backup_heater_active"
]

target = "next_hour_temperature"


# -----------------------------
# 3. Create X and y
# -----------------------------

X = df[features]
y = df[target]

print("\nFeatures:")
print(features)

print(f"\nTarget:")
print(target)

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")


# -----------------------------
# 4. Chronological train/test split
# -----------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTrain/test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -----------------------------
# 5. Save prepared datasets
# -----------------------------

X_train.to_csv(
    "data/synthetic/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/synthetic/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/synthetic/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/synthetic/y_test.csv",
    index=False
)

print("\nPrepared datasets saved successfully!")