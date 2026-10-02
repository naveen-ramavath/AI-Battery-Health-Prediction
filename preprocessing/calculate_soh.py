import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "discharge_data.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "discharge_data_with_soh.csv"


# ============================================================
# NASA rated capacity
# ============================================================

RATED_CAPACITY = 2.0  # Ah


# ============================================================
# Load extracted discharge data
# ============================================================

print("Loading:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nTotal measurement rows:", len(df))


# ============================================================
# Check capacity values
# ============================================================

print("\nCapacity information:")

print("Minimum capacity:", df["capacity"].min())
print("Maximum capacity:", df["capacity"].max())


# ============================================================
# Calculate SOH
# ============================================================

df["SOH"] = df["capacity"] / RATED_CAPACITY


# ============================================================
# Save result
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display summary
# ============================================================

print("\nSOH information:")

print("Minimum SOH:", df["SOH"].min())
print("Maximum SOH:", df["SOH"].max())

print("\nFirst 10 rows:")

print(
    df[
        [
            "battery",
            "cycle_number",
            "capacity",
            "SOH"
        ]
    ].head(10)
)

print("\nSOH calculation complete.")

print("\nSaved to:")
print(OUTPUT_FILE)