import scipy.io
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data" / "raw" / "B0005.mat"

print("Loading:", file_path)

data = scipy.io.loadmat(
    file_path,
    squeeze_me=True,
    struct_as_record=False
)

battery = data["B0005"]
cycles = battery.cycle
print(f"\nTotal cycle records in battery: {len(cycles)}")

# Find and inspect the first discharge cycle
discharge_cycle = None
discharge_index = None

for i, cycle in enumerate(cycles):
    # Cycle types include: 'charge', 'discharge', 'impedance'
    if cycle.type == "discharge":
        discharge_cycle = cycle
        discharge_index = i
        break

print("\n===== FIRST DISCHARGE CYCLE =====")
print("Cycle index:", discharge_index)
print("Type:", discharge_cycle.type)
print("Ambient temperature:", discharge_cycle.ambient_temperature)

print("\nTime information:")
print(discharge_cycle.time)
