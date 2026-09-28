import pandas as pd
import os

# ============================================================
# 1. File location
# ============================================================

DATA_PATH = "data/real/E2025-08-27_2026-08-27.csv"


# ============================================================
# 2. Check that the file exists
# ============================================================

if not os.path.exists(DATA_PATH):
    print("ERROR: Dataset not found.")
    print(f"Expected location: {DATA_PATH}")
    print("\nMake sure the CSV is inside:")
    print("data/real/")
    raise SystemExit


# ============================================================
# 3. Load the dataset
# ============================================================

print("=" * 70)
print("LOADING REAL SOLAR THERMAL DATASET")
print("=" * 70)

# The dataset uses semicolons as separators
df = pd.read_csv(DATA_PATH, sep=";")

print("\nDataset loaded successfully!")


# ============================================================
# 4. Basic dataset information
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 5. Column names
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 6. First five rows
# ============================================================

print("\n" + "=" * 70)
print("3. FIRST FIVE ROWS")
print("=" * 70)

print(df.head().to_string())


# ============================================================
# 7. Data types
# ============================================================

print("\n" + "=" * 70)
print("4. DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())


# ============================================================
# 8. Missing values
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found!")
else:
    print(missing.to_string())


# ============================================================
# 9. Duplicate rows
# ============================================================

print("\n" + "=" * 70)
print("6. DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")


# ============================================================
# 10. Convert timestamp
# ============================================================

print("\n" + "=" * 70)
print("7. TIMESTAMP INFORMATION")
print("=" * 70)

# The timestamp column appears to be named 'time'
if "time" in df.columns:

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    print(f"Start: {df['time'].min()}")
    print(f"End:   {df['time'].max()}")

    invalid_dates = df["time"].isna().sum()

    print(f"Invalid timestamps: {invalid_dates:,}")

else:
    print("WARNING: No column named 'time' was found.")


# ============================================================
# 11. Measurement interval
# ============================================================

print("\n" + "=" * 70)
print("8. MEASUREMENT INTERVAL")
print("=" * 70)

if "time" in df.columns:

    time_difference = (
        df["time"]
        .sort_values()
        .diff()
        .dropna()
    )

    print("Most common time intervals:")

    print(
        time_difference
        .value_counts()
        .head(10)
        .to_string()
    )


# ============================================================
# 12. Numeric summary
# ============================================================

print("\n" + "=" * 70)
print("9. NUMERIC DATA SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    df[numeric_columns]
    .describe()
    .transpose()
    .to_string()
)


# ============================================================
# 13. Identify installation/system prefixes
# ============================================================

print("\n" + "=" * 70)
print("10. INSTALLATIONS FOUND")
print("=" * 70)

installations = set()

for column in df.columns:

    if "_" in column:

        prefix = column.split("_")[0]

        if prefix.isdigit():
            installations.add(prefix)

if installations:

    print(
        "Detected installation IDs:",
        ", ".join(sorted(installations))
    )

else:

    print("No installation IDs automatically detected.")


# ============================================================
# 14. Search for important variables
# ============================================================

print("\n" + "=" * 70)
print("11. IMPORTANT VARIABLES")
print("=" * 70)

keywords = [
    "temp",
    "temperature",
    "solar",
    "radiation",
    "flow",
    "volume",
    "pump",
    "heater",
    "power",
    "energy",
    "ambient"
]

for keyword in keywords:

    matching_columns = [
        column
        for column in df.columns
        if keyword.lower() in column.lower()
    ]

    print(f"\n--- {keyword.upper()} ---")

    if matching_columns:

        for column in matching_columns:
            print(column)

    else:

        print("None found.")


# ============================================================
# 15. Save a simple inspection report
# ============================================================

REPORT_PATH = "data/real/inspection_report.txt"

with open(REPORT_PATH, "w", encoding="utf-8") as report:

    report.write("REAL SOLAR THERMAL DATASET INSPECTION\n")
    report.write("=" * 60 + "\n\n")

    report.write(
        f"Rows: {df.shape[0]:,}\n"
    )

    report.write(
        f"Columns: {df.shape[1]}\n\n"
    )

    report.write("COLUMN NAMES\n")
    report.write("-" * 60 + "\n")

    for column in df.columns:
        report.write(f"{column}\n")

    report.write("\nMISSING VALUES\n")
    report.write("-" * 60 + "\n")

    if len(missing) == 0:
        report.write("No missing values found.\n")
    else:
        report.write(
            missing.to_string()
        )

    report.write("\n\nDATA TYPES\n")
    report.write("-" * 60 + "\n")

    report.write(
        df.dtypes.to_string()
    )

    report.write("\n\nNUMERIC SUMMARY\n")
    report.write("-" * 60 + "\n")

    report.write(
        df[numeric_columns]
        .describe()
        .transpose()
        .to_string()
    )


# ============================================================
# 16. Finished
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"\nInspection report saved to:")
print(REPORT_PATH)