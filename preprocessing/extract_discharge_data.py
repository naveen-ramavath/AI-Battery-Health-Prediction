import scipy.io
import numpy as np
from pathlib import Path
import csv

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NASA batteries used in the base paper
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


def extract_battery(battery_name):
    file_path = RAW_DIR / f"{battery_name}.mat"

    print("\n" + "=" * 60)
    print(f"Processing {battery_name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: {file_path} not found")
        return []

    mat = scipy.io.loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False
    )
    battery = mat[battery_name]
    return battery
