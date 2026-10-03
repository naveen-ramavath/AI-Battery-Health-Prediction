import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("BATTERY HEALTH MODEL RESULT VISUALIZATION")
print("=" * 60)

# Load saved predictions
data = np.load("data/processed/predictions.npz")

soh_actual = data["soh_actual"]
soh_predicted = data["soh_predicted"]

rul_actual = data["rul_actual"]
rul_predicted = data["rul_predicted"]

print("Test samples:", len(soh_actual))


# ============================================================
# 1. ACTUAL VS PREDICTED SOH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(soh_actual, label="Actual SOH")
plt.plot(soh_predicted, label="Predicted SOH")

plt.xlabel("Test Sample")
plt.ylabel("SOH")
plt.title("Actual vs Predicted SOH")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "data/processed/actual_vs_predicted_soh.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. ACTUAL VS PREDICTED RUL
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(rul_actual, label="Actual RUL")
plt.plot(rul_predicted, label="Predicted RUL")

plt.xlabel("Test Sample")
plt.ylabel("RUL (cycles)")
plt.title("Actual vs Predicted RUL")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "data/processed/actual_vs_predicted_rul.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. SOH ERROR
# ============================================================

soh_error = soh_actual - soh_predicted

plt.figure(figsize=(10, 5))

plt.plot(soh_error)
plt.axhline(0, linestyle="--")

plt.xlabel("Test Sample")
plt.ylabel("SOH Error")
plt.title("SOH Prediction Error")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "data/processed/soh_error.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. RUL ERROR
# ============================================================

rul_error = rul_actual - rul_predicted

plt.figure(figsize=(10, 5))

plt.plot(rul_error)
plt.axhline(0, linestyle="--")

plt.xlabel("Test Sample")
plt.ylabel("RUL Error (cycles)")
plt.title("RUL Prediction Error")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "data/processed/rul_error.png",
    dpi=300
)

plt.show()


print()
print("============================================================")
print("VISUALIZATION COMPLETE")
print("============================================================")

print("Saved:")
print("1. actual_vs_predicted_soh.png")
print("2. actual_vs_predicted_rul.png")
print("3. soh_error.png")
print("4. rul_error.png")