import numpy as np

PATH = "data/processed/normalized_fragments.npz"

print("=" * 60)
print("TRAINING DATA VERIFICATION")
print("=" * 60)

data = np.load(PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_soh_train = data["y_soh_train"]
y_soh_test = data["y_soh_test"]

y_rul_train = data["y_rul_train"]
y_rul_test = data["y_rul_test"]

print("\nTraining data:")
print("X_train:", X_train.shape)
print("SOH_train:", y_soh_train.shape)
print("RUL_train:", y_rul_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("SOH_test:", y_soh_test.shape)
print("RUL_test:", y_rul_test.shape)

# ---------------------------------------------------------
# Shape verification
# ---------------------------------------------------------

assert len(X_train) == len(y_soh_train)
assert len(X_train) == len(y_rul_train)

assert len(X_test) == len(y_soh_test)
assert len(X_test) == len(y_rul_test)

print("\n✓ Training input and labels match")
print("✓ Testing input and labels match")

# ---------------------------------------------------------
# Check for NaN / infinity
# ---------------------------------------------------------

print("\nChecking invalid values...")

print("X_train NaN:", np.isnan(X_train).any())
print("X_train Inf:", np.isinf(X_train).any())

print("SOH NaN:", np.isnan(y_soh_train).any())
print("SOH Inf:", np.isinf(y_soh_train).any())

print("RUL NaN:", np.isnan(y_rul_train).any())
print("RUL Inf:", np.isinf(y_rul_train).any())

# ---------------------------------------------------------
# Label ranges
# ---------------------------------------------------------

print("\nSOH range:")
print("Minimum:", y_soh_train.min())
print("Maximum:", y_soh_train.max())

print("\nRUL range:")
print("Minimum:", y_rul_train.min())
print("Maximum:", y_rul_train.max())

# ---------------------------------------------------------
# First few labels
# ---------------------------------------------------------

print("\nFirst 10 training labels:")

for i in range(min(10, len(X_train))):

    print(
        f"{i}: "
        f"SOH={y_soh_train[i]:.6f}, "
        f"RUL={y_rul_train[i]:.0f}"
    )

print("\n" + "=" * 60)
print("TRAINING DATA VERIFICATION COMPLETE")
print("=" * 60)