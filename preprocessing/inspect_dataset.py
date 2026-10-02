import scipy.io
from pathlib import Path

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

# Find the first discharge cycle
discharge_cycle = None
discharge_index = None

for i, cycle in enumerate(cycles):
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

print("\n===== DISCHARGE DATA =====")

cycle_data = discharge_cycle.data

for field in cycle_data._fieldnames:
    value = getattr(cycle_data, field)

    print(f"\nField: {field}")
    print("Shape:", getattr(value, "shape", "No shape"))

    # Show first 5 values for numerical arrays
    if hasattr(value, "shape") and value.size > 0:
        print("First 5 values:", value[:5])