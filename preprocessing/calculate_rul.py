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
# (B0007 did not reach the 70% SOH threshold and is excluded)
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
