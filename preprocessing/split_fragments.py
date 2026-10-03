import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fragments.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_test_fragments.npz"
)


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.30
RANDOM_STATE = 42


# ============================================================
# Load fragments
# ============================================================

print("=" * 60)
print("LOADING FRAGMENTS")
print("=" * 60)

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

X = data["X"]
y_soh = data["y_soh"]
y_rul = data["y_rul"]
fragment_info = data["fragment_info"]

print("Total fragments:", len(X))
print("Input shape:", X.shape)
print("SOH shape:", y_soh.shape)
print("RUL shape:", y_rul.shape)


# ============================================================
# Create indices
# ============================================================

indices = np.arange(len(X))


# ============================================================
# 70% Training / 30% Testing
# ============================================================

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)


# ============================================================
# Create datasets
# ============================================================

X_train = X[train_indices]
X_test = X[test_indices]

y_soh_train = y_soh[train_indices]
y_soh_test = y_soh[test_indices]

y_rul_train = y_rul[train_indices]
y_rul_test = y_rul[test_indices]

info_train = fragment_info[train_indices]
info_test = fragment_info[test_indices]


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPLETE")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining input shape:")
print(X_train.shape)

print("\nTesting input shape:")
print(X_test.shape)

print("\nTraining SOH shape:")
print(y_soh_train.shape)

print("Testing SOH shape:")
print(y_soh_test.shape)

print("\nTraining RUL shape:")
print(y_rul_train.shape)

print("Testing RUL shape:")
print(y_rul_test.shape)


# ============================================================
# Save
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train,
    X_test=X_test,

    y_soh_train=y_soh_train,
    y_soh_test=y_soh_test,

    y_rul_train=y_rul_train,
    y_rul_test=y_rul_test,

    info_train=info_train,
    info_test=info_test
)

print("\nSaved to:")
print(OUTPUT_FILE)