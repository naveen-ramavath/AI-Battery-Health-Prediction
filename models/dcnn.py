import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size):

        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Save original input
        residual = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out = out + residual

        out = self.relu(out)

        return out


class DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Input:
        # (batch, 128, 2)

        # Convert 2 input features
        # into 64 feature channels.

        self.input_conv = nn.Conv1d(
            in_channels=2,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # Four convolution/residual stages
        # using the paper's kernel sizes.

        self.block1 = ResidualBlock(
            channels=64,
            kernel_size=3
        )

        self.block2 = ResidualBlock(
            channels=64,
            kernel_size=5
        )

        self.block3 = ResidualBlock(
            channels=64,
            kernel_size=7
        )

        self.block4 = ResidualBlock(
            channels=64,
            kernel_size=9
        )

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        # Input:
        # (batch, 128, 2)

        # Conv1D expects:
        # (batch, channels, sequence)

        x = x.transpose(1, 2)

        # Initial feature extraction
        x = self.input_conv(x)
        x = self.relu(x)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Max pooling
        x = self.pool(x)

        return x


# ============================================================
# Test DCNN
# ============================================================

if __name__ == "__main__":

    model = DCNN()

    test_input = torch.randn(
        4,
        128,
        2
    )

    output = model(test_input)

    print("=" * 60)
    print("DCNN RESIDUAL TEST")
    print("=" * 60)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

    print("\nDCNN architecture:")
    print(model)