import torch
import torch.nn as nn

from models.dcnn_lstm import DCNN_LSTM


class DCNN_LSTM_MHA(nn.Module):

    def __init__(self):

        super().__init__()

        # DCNN + LSTM feature extractor
        self.feature_extractor = DCNN_LSTM()

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True
        )

        # Normalization
        self.norm = nn.LayerNorm(128)

    def forward(self, x):

        # ------------------------------------------------
        # Input:
        # (batch, 128, 2)
        # ------------------------------------------------

        x = self.feature_extractor(x)

        # LSTM output:
        # (batch, 64, 64)

        # Multi-Head Attention
        attention_output, attention_weights = self.attention(
            x,
            x,
            x
        )

        # Residual connection
        x = x + attention_output

        # Layer normalization
        x = self.norm(x)

        return x, attention_weights


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = DCNN_LSTM_MHA()

    # Temporary test batch
    test_input = torch.randn(
        4,
        128,
        2
    )

    output, attention_weights = model(test_input)

    print("=" * 60)
    print("DCNN + LSTM + MHA TEST")
    print("=" * 60)

    print("Input shape:")
    print(test_input.shape)

    print("\nAttention output shape:")
    print(output.shape)

    print("\nAttention weights shape:")
    print(attention_weights.shape)

    print("\nModel:")
    print(model)