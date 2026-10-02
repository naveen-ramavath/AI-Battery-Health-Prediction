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

# 70% SOH corresponds to 1.4 Ah capacity threshold
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
