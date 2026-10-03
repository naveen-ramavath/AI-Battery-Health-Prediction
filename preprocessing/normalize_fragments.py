import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_test_fragments.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized_fragments.npz"
)


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("LOADING TRAIN / TEST DATA")
print("=" * 60)

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

X_train = data["X_train"]
X_test = data["X_test"]

y_soh_train = data["y_soh_train"]
y_soh_test = data["y_soh_test"]

y_rul_train = data["y_rul_train"]
y_rul_test = data["y_rul_test"]

info_train = data["info_train"]
info_test = data["info_test"]

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ============================================================
# Calculate training mean and standard deviation
# ============================================================

train_mean = X_train.mean(axis=(0, 1))
train_std = X_train.std(axis=(0, 1))

# Prevent division by zero
train_std[train_std == 0] = 1.0


print("\nTraining feature statistics:")

print("\nVoltage:")
print("Mean:", train_mean[0])
print("Std :", train_std[0])

print("\nCurrent:")
print("Mean:", train_mean[1])
print("Std :", train_std[1])


# ============================================================
# Normalize
# ============================================================

X_train_normalized = (
    X_train - train_mean
) / train_std

X_test_normalized = (
    X_test - train_mean
) / train_std


# ============================================================
# Check normalized data
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZATION COMPLETE")
print("=" * 60)

print("\nNormalized training shape:")
print(X_train_normalized.shape)

print("\nNormalized testing shape:")
print(X_test_normalized.shape)

print("\nNormalized training mean:")
print(X_train_normalized.mean(axis=(0, 1)))

print("\nNormalized training std:")
print(X_train_normalized.std(axis=(0, 1)))


# ============================================================
# Save
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,

    X_train=X_train_normalized,
    X_test=X_test_normalized,

    y_soh_train=y_soh_train,
    y_soh_test=y_soh_test,

    y_rul_train=y_rul_train,
    y_rul_test=y_rul_test,

    info_train=info_train,
    info_test=info_test,

    train_mean=train_mean,
    train_std=train_std
)

print("\nSaved to:")
print(OUTPUT_FILE)