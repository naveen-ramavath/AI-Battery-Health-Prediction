import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "discharge_data_with_soh.csv"
)

EOL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eol_information.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "discharge_data_with_soh_rul.csv"
)


# ============================================================
# Load data
# ============================================================

print("Loading discharge data...")

df = pd.read_csv(INPUT_FILE)

print("Total measurement rows:", len(df))


print("\nLoading EOL information...")

eol_df = pd.read_csv(EOL_FILE)

print(eol_df)


# ============================================================
# Keep only batteries for which EOL was actually identified
# ============================================================

valid_eol = eol_df[
    eol_df["eol_discharge_number"].notna()
].copy()


print("\nBatteries with valid EOL:")

print(
    valid_eol[
        [
            "battery",
            "eol_discharge_number",
            "eol_capacity",
            "eol_soh"
        ]
    ]
)


# ============================================================
# Create one row per discharge cycle
# ============================================================

cycle_df = (
    df[
        [
            "battery",
            "cycle_number",
            "discharge_number",
            "capacity",
            "SOH"
        ]
    ]
    .drop_duplicates(
        subset=["battery", "discharge_number"]
    )
    .copy()
)


# ============================================================
# Calculate RUL
# ============================================================

cycle_results = []

for _, eol in valid_eol.iterrows():

    battery = eol["battery"]

    eol_discharge = int(
        eol["eol_discharge_number"]
    )

    battery_cycles = cycle_df[
        cycle_df["battery"] == battery
    ].copy()

    # Keep cycles up to and including EOL
    battery_cycles = battery_cycles[
        battery_cycles["discharge_number"]
        <= eol_discharge
    ].copy()

    # RUL = EOL cycle - current cycle
    battery_cycles["RUL"] = (
        eol_discharge
        - battery_cycles["discharge_number"]
    )

    cycle_results.append(battery_cycles)


# ============================================================
# Combine batteries
# ============================================================

rul_cycles = pd.concat(
    cycle_results,
    ignore_index=True
)


# ============================================================
# Add RUL to original measurement rows
# ============================================================

df = df.merge(
    rul_cycles[
        [
            "battery",
            "discharge_number",
            "RUL"
        ]
    ],
    on=[
        "battery",
        "discharge_number"
    ],
    how="inner"
)


# ============================================================
# Save final dataset
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display summary
# ============================================================

print("\n" + "=" * 60)
print("RUL CALCULATION COMPLETE")
print("=" * 60)

print("\nFinal measurement rows:", len(df))

print("\nBatteries included:")

print(
    df["battery"]
    .value_counts()
    .sort_index()
)


print("\nRUL range:")

print("Minimum RUL:", df["RUL"].min())
print("Maximum RUL:", df["RUL"].max())


print("\nExample cycle-level RUL values:")

example = (
    rul_cycles
    .groupby("battery")
    .head(5)
)

print(
    example[
        [
            "battery",
            "discharge_number",
            "capacity",
            "SOH",
            "RUL"
        ]
    ]
)


print("\nLast cycles before EOL:")

last_cycles = (
    rul_cycles
    .sort_values(
        ["battery", "discharge_number"]
    )
    .groupby("battery")
    .tail(3)
)

print(
    last_cycles[
        [
            "battery",
            "discharge_number",
            "capacity",
            "SOH",
            "RUL"
        ]
    ]
)


print("\nSaved to:")
print(OUTPUT_FILE)