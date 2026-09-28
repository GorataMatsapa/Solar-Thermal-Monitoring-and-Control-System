import numpy as np
import pandas as pd

# -----------------------------
# 1. Simulation configuration
# -----------------------------

DAYS = 365
HOURS = DAYS * 24

np.random.seed(42)

# -----------------------------
# 2. Time information
# -----------------------------

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=HOURS,
    freq="h"
)

df = pd.DataFrame({
    "timestamp": timestamps
})

df["hour"] = df["timestamp"].dt.hour
df["day_of_year"] = df["timestamp"].dt.dayofyear


# -----------------------------
# 3. Simulate ambient temperature
# -----------------------------

# Seasonal variation
seasonal_temperature = 7 * np.sin(
    2 * np.pi * (df["day_of_year"] - 80) / 365
)

# Daily variation
daily_temperature = 5 * np.sin(
    2 * np.pi * (df["hour"] - 8) / 24
)

# Random weather variation
weather_noise = np.random.normal(0, 1.5, HOURS)

df["ambient_temperature"] = (
    24
    + seasonal_temperature
    + daily_temperature
    + weather_noise
)


# -----------------------------
# 4. Simulate solar irradiance
# -----------------------------

# Solar radiation is zero at night
solar_angle = np.sin(
    np.pi * (df["hour"] - 6) / 12
)

solar_angle = np.maximum(solar_angle, 0)

# Seasonal solar variation
seasonal_solar = (
    0.85
    + 0.15 * np.sin(
        2 * np.pi * (df["day_of_year"] - 80) / 365
    )
)

solar_noise = np.random.normal(0, 50, HOURS)

df["solar_irradiance"] = (
    900
    * solar_angle
    * seasonal_solar
    + solar_noise
)

df["solar_irradiance"] = np.maximum(
    df["solar_irradiance"], 0
)


# -----------------------------
# 5. Simulate water demand
# -----------------------------

df["flow_rate"] = 0.0

# Morning demand
morning_demand = (
    (df["hour"] >= 6) &
    (df["hour"] <= 9)
)

# Evening demand
evening_demand = (
    (df["hour"] >= 18) &
    (df["hour"] <= 21)
)

df.loc[morning_demand, "flow_rate"] = np.random.uniform(
    0, 4, morning_demand.sum()
)

df.loc[evening_demand, "flow_rate"] = np.random.uniform(
    0, 5, evening_demand.sum()
)


# -----------------------------
# 6. Simulate tank temperature (With Look-Ahead Control)
# -----------------------------

water_temperature = np.zeros(HOURS)
backup_heater_status = np.zeros(HOURS)  # 0 = Off, 1 = On

water_temperature[0] = 30.0

tank_mass = 300.0          
specific_heat = 4.186      
collector_area = 4.0       
collector_efficiency = 0.40
heat_loss_coefficient = 0.025
heater_power_kw = 3.0      # 3 kW backup electric element

for i in range(1, HOURS):
    previous_temperature = water_temperature[i - 1]
    
    # Using .iloc for positional safety
    ambient = df.iloc[i]["ambient_temperature"]
    irradiance = df.iloc[i]["solar_irradiance"]
    flow = df.iloc[i]["flow_rate"]
    hour = df.iloc[i]["hour"]

    # Solar energy contribution
    solar_heat = (irradiance * collector_area * collector_efficiency / (tank_mass * specific_heat)) * 3.6

    # Heat loss to environment
    heat_loss = heat_loss_coefficient * (previous_temperature - ambient)

    # Temperature reduction from hot-water demand
    draw_loss = flow * 0.08

    # --- LOOK-AHEAD CONTROL LOGIC ---
    # Anticipate heavy morning draw (6 AM - 9 AM) or evening draw (6 PM - 10 PM)
    # If temp is heading low right before peak hours, kick on the 3kW heater
    is_pre_peak = (hour == 5) or (hour == 17)
    if is_pre_peak and (previous_temperature < 45.0):
        backup_active = 1
        # 3 kW element running for 1 hour = 3000 W * 3600 s = 10800 kJ
        heater_heat = (heater_power_kw * 3600) / (tank_mass * specific_heat)
    else:
        backup_active = 0
        heater_heat = 0.0
        
    backup_heater_status[i] = backup_active
    system_noise = np.random.normal(0, 0.08)

    # Thermal evolution calculation
    new_temperature = (
        previous_temperature
        + solar_heat
        + heater_heat  # Injected predictive control element
        - heat_loss
        - draw_loss
        + system_noise
    )

    new_temperature = np.clip(new_temperature, 15, 80)
    water_temperature[i] = new_temperature

df["water_temperature"] = water_temperature
df["backup_heater_active"] = backup_heater_status



# -----------------------------
# 7. Create prediction target
# -----------------------------

df["next_hour_temperature"] = (
    df["water_temperature"].shift(-1)
)

# Remove final row because it has no next-hour value
df = df.dropna()


# -----------------------------
# 8. Save dataset
# -----------------------------

output_path = (
    "data/synthetic/"
    "synthetic_solar_thermal_data.csv"
)

df.to_csv(output_path, index=False)

print("Synthetic dataset generated successfully!")
print(f"Rows: {len(df)}")
print(f"Saved to: {output_path}")

print("\nFirst five rows:")
print(df.head())