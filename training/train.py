import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.prediction_heads import BatteryHealthModel


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/normalized_fragments.npz"
MODEL_PATH = "data/processed/battery_health_model.pth"

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("BATTERY HEALTH MODEL TRAINING")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_soh_train = data["y_soh_train"]
y_soh_test = data["y_soh_test"]

y_rul_train = data["y_rul_train"]
y_rul_test = data["y_rul_test"]

print("Training input:", X_train.shape)
print("Training SOH:", y_soh_train.shape)
print("Training RUL:", y_rul_train.shape)

print("Testing input:", X_test.shape)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_soh_train = torch.tensor(
    y_soh_train,
    dtype=torch.float32
).unsqueeze(1)

y_rul_train = torch.tensor(
    y_rul_train,
    dtype=torch.float32
).unsqueeze(1)


# ============================================================
# DATASET + DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_soh_train,
    y_rul_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# MODEL
# ============================================================

model = BatteryHealthModel().to(device)

print("\nModel loaded successfully.")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

soh_loss_function = nn.MSELoss()
rul_loss_function = nn.MSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_soh_loss = 0.0
    total_rul_loss = 0.0

    for X_batch, soh_batch, rul_batch in train_loader:

        X_batch = X_batch.to(device)
        soh_batch = soh_batch.to(device)
        rul_batch = rul_batch.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predicted_soh, predicted_rul, _ = model(X_batch)

        # ----------------------------------------------------
        # Calculate losses
        # ----------------------------------------------------

        soh_loss = soh_loss_function(
            predicted_soh,
            soh_batch
        )

        rul_loss = rul_loss_function(
            predicted_rul,
            rul_batch
        )

        # Combined loss
        loss = soh_loss + rul_loss

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        total_loss += loss.item()
        total_soh_loss += soh_loss.item()
        total_rul_loss += rul_loss.item()

    # Average losses
    batches = len(train_loader)

    avg_loss = total_loss / batches
    avg_soh_loss = total_soh_loss / batches
    avg_rul_loss = total_rul_loss / batches

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f} | "
        f"SOH Loss: {avg_soh_loss:.6f} | "
        f"RUL Loss: {avg_rul_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel saved to:")
print(MODEL_PATH)