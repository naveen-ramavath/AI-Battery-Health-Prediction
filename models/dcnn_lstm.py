import torch
import torch.nn as nn

from models.dcnn import DCNN


class DCNN_LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        # DCNN feature extractor
        self.dcnn = DCNN()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):

        # ------------------------------------------------
        # Input
        # (batch, 128, 2)
        # ------------------------------------------------

        x = self.dcnn(x)

        # DCNN output:
        # (batch, 64, 64)
        #
        # PyTorch DCNN format:
        # (batch, channels, sequence)
        #
        # LSTM format:
        # (batch, sequence, features)

        x = x.transpose(1, 2)

        # Now:
        # (batch, 64, 64)

        # LSTM
        x, _ = self.lstm(x)

        return x


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = DCNN_LSTM()

    # 4 fake fragments for testing
    test_input = torch.randn(4, 128, 2)

    output = model(test_input)

    print("=" * 60)
    print("DCNN + LSTM TEST")
    print("=" * 60)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

    print("\nModel:")
    print(model)