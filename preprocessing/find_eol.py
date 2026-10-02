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


# ============================================================
# NASA battery settings from the base paper
# ============================================================

RATED_CAPACITY = 2.0       # Ah
EOL_SOH = 0.70

EOL_CAPACITY = RATED_CAPACITY * EOL_SOH


print("Loading:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)


# ============================================================
# Get one row per discharge cycle
# ============================================================

cycles = (
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
    .sort_values(
        ["battery", "discharge_number"]
    )
)


print("\n" + "=" * 60)
print("EOL ANALYSIS")
print("=" * 60)

print(f"Rated capacity : {RATED_CAPACITY} Ah")
print(f"EOL SOH        : {EOL_SOH}")
print(f"EOL capacity   : {EOL_CAPACITY} Ah")


# ============================================================
# Find EOL for each battery
# ============================================================

eol_results = []

for battery in sorted(cycles["battery"].unique()):

    battery_data = cycles[
        cycles["battery"] == battery
    ].copy()

    # First discharge cycle reaching the EOL threshold
    eol_rows = battery_data[
        battery_data["capacity"] <= EOL_CAPACITY
    ]

    print("\n" + "-" * 60)
    print(f"Battery: {battery}")

    if len(eol_rows) == 0:

        print("EOL threshold was NOT reached.")

        print(
            "Minimum capacity:",
            battery_data["capacity"].min()
        )

        print(
            "Minimum SOH:",
            battery_data["SOH"].min()
        )

        continue

    eol = eol_rows.iloc[0]

    print("EOL discharge number:",
          int(eol["discharge_number"]))

    print("Original cycle number:",
          int(eol["cycle_number"]))

    print("EOL capacity:",
          eol["capacity"])

    print("EOL SOH:",
          eol["SOH"])

    eol_results.append({
        "battery": battery,
        "eol_discharge_number":
            int(eol["discharge_number"]),
        "eol_cycle_number":
            int(eol["cycle_number"]),
        "eol_capacity":
            float(eol["capacity"]),
        "eol_soh":
            float(eol["SOH"])
    })


# ============================================================
# Save EOL information
# ============================================================

eol_df = pd.DataFrame(eol_results)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eol_information.csv"
)

eol_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("EOL ANALYSIS COMPLETE")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_FILE)