import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "discharge_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "discharge_data_with_soh.csv"

# NASA rated capacity
RATED_CAPACITY = 2.0  # Ah

print("Loading:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE)
print("\nTotal measurement rows:", len(df))

# Calculate SOH using current capacity divided by rated 2 Ah capacity
df["SOH"] = df["capacity"] / RATED_CAPACITY
