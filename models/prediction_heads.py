import torch
import torch.nn as nn

from models.mha import DCNN_LSTM_MHA


class BatteryHealthModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature extraction:
        # DCNN → LSTM → MHA
        self.feature_extractor = DCNN_LSTM_MHA()

        # Convert the sequence representation
        # into one feature vector.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # SOH prediction head
        self.soh_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # RUL prediction head
        self.rul_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        # ------------------------------------------------
        # DCNN → LSTM → MHA
        # ------------------------------------------------

        features, attention_weights = self.feature_extractor(x)

        # features:
        # (batch, 64, 64)

        # Convert:
        # (batch, sequence, features)
        #
        # to:
        # (batch, features, sequence)

        features = features.transpose(1, 2)

        # Global average pooling
        features = self.global_pool(features)

        # Remove final dimension
        features = features.squeeze(-1)

        # Now:
        # (batch, 64)

        # ------------------------------------------------
        # Predictions
        # ------------------------------------------------

        soh = self.soh_head(features)

        rul = self.rul_head(features)

        return soh, rul, attention_weights


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = BatteryHealthModel()

    # Temporary test batch
    test_input = torch.randn(
        4,
        128,
        2
    )

    soh, rul, attention = model(test_input)

    print("=" * 60)
    print("COMPLETE PREDICTION HEAD TEST")
    print("=" * 60)

    print("Input shape:")
    print(test_input.shape)

    print("\nSOH prediction shape:")
    print(soh.shape)

    print("\nRUL prediction shape:")
    print(rul.shape)

    print("\nAttention shape:")
    print(attention.shape)

    print("\nExample SOH predictions:")
    print(soh.detach().numpy().flatten())

    print("\nExample RUL predictions:")
    print(rul.detach().numpy().flatten())