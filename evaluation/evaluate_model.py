import numpy as np
import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction_heads import BatteryHealthModel


print("=" * 60)
print("BATTERY HEALTH MODEL EVALUATION")
print("=" * 60)

# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

data_path = "data/processed/normalized_fragments.npz"

print("\nLoading test data...")

data = np.load(data_path)

X_test = data["X_test"]
SOH_test = data["y_soh_test"]
RUL_test = data["y_rul_test"]

print("Test input:", X_test.shape)
print("Test SOH:", SOH_test.shape)
print("Test RUL:", RUL_test.shape)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

device = torch.device("cpu")

model = BatteryHealthModel()

model_path = "data/processed/battery_health_model.pth"

print("\nLoading trained model...")

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 3. Convert test data to tensors
# --------------------------------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


# --------------------------------------------------
# 4. Make predictions
# --------------------------------------------------

print("\nGenerating predictions...")

with torch.no_grad():

    soh_pred, rul_pred, attention = model(
        X_test_tensor
    )

soh_pred = soh_pred.cpu().numpy().flatten()
rul_pred = rul_pred.cpu().numpy().flatten()


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

print("\nNumber of test samples:", len(X_test))

print("\nFirst 10 SOH predictions:")

for i in range(min(10, len(X_test))):

    print(
        f"{i}: "
        f"Actual SOH = {SOH_test[i]:.6f} | "
        f"Predicted SOH = {soh_pred[i]:.6f}"
    )


print("\nFirst 10 RUL predictions:")

for i in range(min(10, len(X_test))):

    print(
        f"{i}: "
        f"Actual RUL = {RUL_test[i]:.2f} | "
        f"Predicted RUL = {rul_pred[i]:.2f}"
    )


# --------------------------------------------------
# 6. Prediction ranges
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION RANGES")
print("=" * 60)

print("\nSOH:")
print("Actual minimum :", SOH_test.min())
print("Actual maximum :", SOH_test.max())
print("Predicted minimum:", soh_pred.min())
print("Predicted maximum:", soh_pred.max())

print("\nRUL:")
print("Actual minimum :", RUL_test.min())
print("Actual maximum :", RUL_test.max())
print("Predicted minimum:", rul_pred.min())
print("Predicted maximum:", rul_pred.max())


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)


# ============================================================
# SAVE PREDICTIONS FOR VISUALIZATION
# ============================================================

np.savez(
    "data/processed/predictions.npz",
    soh_actual=SOH_test,
    soh_predicted=soh_pred,
    rul_actual=RUL_test,
    rul_predicted=rul_pred
)

print()
print("Predictions saved to:")
print("data/processed/predictions.npz")