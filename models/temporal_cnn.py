"""
Temporal CNN Module

Extracts temporal features from multivariate time series data.

Input shape:
    (batch_size, channels, time_steps)

Example:
    (32, 6, 128)

Output shape:
    (batch_size, channels, feature_dim)

This module learns temporal patterns for each sensor channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network for extracting temporal
    features from multivariate time series data.
    """

    def __init__(self, in_channels=6, hidden_channels=32, out_channels=64, kernel_size=3, dropout=0.3):
        super(TemporalCNN, self).__init__()

        padding = kernel_size // 2

        # First temporal convolution
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels,kernel_size=kernel_size,padding=padding)

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Second temporal convolution
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        # Third temporal convolution
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.bn3 = nn.BatchNorm1d(out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Global pooling to compress time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dropout(x)

        # Global average pooling over time
        x = self.global_pool(x)

        # remove last dimension
        x = x.squeeze(-1)

        return x


if __name__ == "__main__":

    # Example test run
    batch_size = 32
    channels = 6
    time_steps = 128

    dummy_input = torch.randn(batch_size, channels, time_steps)

    model = TemporalCNN()

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)