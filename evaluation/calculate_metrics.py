import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("BATTERY HEALTH MODEL METRICS")
print("=" * 60)

# Load saved predictions if available
# For now, reproduce predictions from evaluation script

import torch
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.prediction_heads import BatteryHealthModel


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

data = np.load(
    "data/processed/normalized_fragments.npz"
)

X_test = data["X_test"]
SOH_test = data["y_soh_test"]
RUL_test = data["y_rul_test"]


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

device = torch.device("cpu")

model = BatteryHealthModel()

model.load_state_dict(
    torch.load(
        "data/processed/battery_health_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# --------------------------------------------------
# 3. Generate predictions
# --------------------------------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    soh_pred, rul_pred, attention = model(
        X_test_tensor
    )

soh_pred = soh_pred.cpu().numpy().flatten()
rul_pred = rul_pred.cpu().numpy().flatten()


# --------------------------------------------------
# 4. SOH metrics
# --------------------------------------------------

soh_mae = mean_absolute_error(
    SOH_test,
    soh_pred
)

soh_rmse = np.sqrt(
    mean_squared_error(
        SOH_test,
        soh_pred
    )
)

soh_r2 = r2_score(
    SOH_test,
    soh_pred
)


# --------------------------------------------------
# 5. RUL metrics
# --------------------------------------------------

rul_mae = mean_absolute_error(
    RUL_test,
    rul_pred
)

rul_rmse = np.sqrt(
    mean_squared_error(
        RUL_test,
        rul_pred
    )
)

rul_r2 = r2_score(
    RUL_test,
    rul_pred
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("SOH PERFORMANCE")
print("=" * 60)

print(f"MAE  : {soh_mae:.6f}")
print(f"RMSE : {soh_rmse:.6f}")
print(f"R²   : {soh_r2:.6f}")


print("\n" + "=" * 60)
print("RUL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {rul_mae:.6f}")
print(f"RMSE : {rul_rmse:.6f}")
print(f"R²   : {rul_r2:.6f}")


print("\n" + "=" * 60)
print("METRICS CALCULATION COMPLETE")
print("=" * 60)