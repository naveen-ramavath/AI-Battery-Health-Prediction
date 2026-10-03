import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "discharge_data_with_soh_rul.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fragments.npz"
)


# ============================================================
# Configuration
# ============================================================

FRAGMENT_LENGTH = 128


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))


# ============================================================
# Check required columns
# ============================================================

required_columns = [
    "battery",
    "discharge_number",
    "voltage",
    "current",
    "SOH",
    "RUL"
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# Create fragments
# ============================================================

X = []
y_soh = []
y_rul = []

fragment_info = []

print("\nCreating fragments...")

for (battery, discharge_number), group in df.groupby(
    ["battery", "discharge_number"]
):

    group = group.reset_index(drop=True)

    voltage = group["voltage"].values
    current = group["current"].values

    soh = group["SOH"].iloc[0]
    rul = group["RUL"].iloc[0]

    # --------------------------------------------------------
    # Skip discharge cycles shorter than 128 measurements
    # --------------------------------------------------------

    if len(group) < FRAGMENT_LENGTH:
        continue

    # --------------------------------------------------------
    # Create non-overlapping 128-point fragments
    # --------------------------------------------------------

    number_of_fragments = (
        len(group) // FRAGMENT_LENGTH
    )

    for fragment_number in range(number_of_fragments):

        start = (
            fragment_number
            * FRAGMENT_LENGTH
        )

        end = start + FRAGMENT_LENGTH

        voltage_fragment = voltage[start:end]
        current_fragment = current[start:end]

        # Shape:
        # (128, 2)
        fragment = np.column_stack(
            (
                voltage_fragment,
                current_fragment
            )
        )

        X.append(fragment)

        y_soh.append(soh)
        y_rul.append(rul)

        fragment_info.append(
            (
                battery,
                discharge_number,
                fragment_number
            )
        )


# ============================================================
# Convert to NumPy arrays
# ============================================================

X = np.array(X, dtype=np.float32)

y_soh = np.array(
    y_soh,
    dtype=np.float32
)

y_rul = np.array(
    y_rul,
    dtype=np.float32
)

fragment_info = np.array(
    fragment_info,
    dtype=object
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("FRAGMENT GENERATION COMPLETE")
print("=" * 60)

print("\nInput fragment shape:")
print(X.shape)

print("\nSOH labels shape:")
print(y_soh.shape)

print("\nRUL labels shape:")
print(y_rul.shape)


# ============================================================
# Show first fragment
# ============================================================

if len(X) > 0:

    print("\nFirst fragment:")
    print(X[0])

    print("\nFirst fragment shape:")
    print(X[0].shape)

    print("\nFirst fragment labels:")
    print("SOH:", y_soh[0])
    print("RUL:", y_rul[0])

    print("\nFirst fragment information:")
    print("Battery:", fragment_info[0][0])
    print("Discharge:", fragment_info[0][1])
    print("Fragment:", fragment_info[0][2])


# ============================================================
# Save
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y_soh=y_soh,
    y_rul=y_rul,
    fragment_info=fragment_info
)

print("\nSaved to:")
print(OUTPUT_FILE)