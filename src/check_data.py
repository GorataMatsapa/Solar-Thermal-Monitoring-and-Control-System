import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/synthetic/synthetic_solar_thermal_data.csv"
)

plt.figure(figsize=(12, 5))

plt.plot(
    df["timestamp"].iloc[:168],
    df["water_temperature"].iloc[:168],
    label="Water Temperature"
)

plt.plot(
    df["timestamp"].iloc[:168],
    df["ambient_temperature"].iloc[:168],
    label="Ambient Temperature"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Synthetic Solar Thermal System - First 7 Days")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/synthetic/thermal_check.png", dpi=150)
print("Graph saved to data/synthetic/thermal_check.png")