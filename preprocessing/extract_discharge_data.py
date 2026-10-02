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


# ============================================================
# Extract discharge data from one battery
# ============================================================

def extract_battery(battery_name):

    file_path = RAW_DIR / f"{battery_name}.mat"

    print("\n" + "=" * 60)
    print(f"Processing {battery_name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: {file_path} not found")
        return []

    # Load MATLAB file
    mat = scipy.io.loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False
    )

    battery = mat[battery_name]

    cycles = battery.cycle

    discharge_records = []

    discharge_number = 0

    for cycle_index, cycle in enumerate(cycles, start=1):

        # Only use discharge cycles
        if cycle.type != "discharge":
            continue

        discharge_number += 1

        cycle_data = cycle.data

        voltage = np.asarray(
            cycle_data.Voltage_measured,
            dtype=float
        )

        current = np.asarray(
            cycle_data.Current_measured,
            dtype=float
        )

        time = np.asarray(
            cycle_data.Time,
            dtype=float
        )

        capacity = float(cycle_data.Capacity)

        # Make sure voltage/current/time have same length
        min_length = min(
            len(voltage),
            len(current),
            len(time)
        )

        voltage = voltage[:min_length]
        current = current[:min_length]
        time = time[:min_length]

        for i in range(min_length):

            discharge_records.append({
                "battery": battery_name,
                "cycle_number": cycle_index,
                "discharge_number": discharge_number,
                "sample_index": i,
                "voltage": voltage[i],
                "current": current[i],
                "time": time[i],
                "capacity": capacity
            })

    print(f"Total cycle records: {len(cycles)}")
    print(f"Discharge cycles: {discharge_number}")
    print(f"Measurement rows: {len(discharge_records)}")

    return discharge_records


# ============================================================
# Process all batteries
# ============================================================

all_records = []

for battery in BATTERIES:

    records = extract_battery(battery)

    all_records.extend(records)


# ============================================================
# Save CSV
# ============================================================

output_file = OUTPUT_DIR / "discharge_data.csv"

fieldnames = [
    "battery",
    "cycle_number",
    "discharge_number",
    "sample_index",
    "voltage",
    "current",
    "time",
    "capacity"
]

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_records)


print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print("Total discharge measurement rows:", len(all_records))
print("Saved to:")
print(output_file)