import torch
from torch import nn


class TemporalConvEncoder(nn.Module):
    """Channel-wise temporal feature extractor used by TERSE."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        c_in = configs.input_channels
        c_mid = configs.mid_channels
        c_out = configs.final_out_channels

        self.block1 = nn.Sequential(
            nn.Conv1d(
                c_in,
                c_mid * c_in,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                padding=configs.kernel_size // 2,
                bias=False,
                groups=c_in,
            ),
            nn.BatchNorm1d(c_mid * c_in),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(
                c_mid * c_in,
                c_mid * 2 * c_in,
                kernel_size=8,
                stride=1,
                padding=4,
                bias=False,
                groups=c_mid * c_in,
            ),
            nn.BatchNorm1d(c_mid * 2 * c_in),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(
                c_mid * 2 * c_in,
                c_out * c_in,
                kernel_size=8,
                stride=1,
                padding=4,
                bias=False,
                groups=c_mid * c_in,
            ),
            nn.BatchNorm1d(c_out * c_in),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return x.view(x.shape[0], self.configs.input_channels, -1)


class TemporalRestorationNet(nn.Module):
    """LSTM restoration head in latent space (temporal restoration task)."""

    def __init__(self, configs):
        super().__init__()
        self.num_channels = configs.input_channels
        self.hidden_dim = configs.AR_hid_dim
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1, self.num_channels)
        out, _ = self.rnn(x)
        return out.view(x.size(0), self.num_channels, -1)
