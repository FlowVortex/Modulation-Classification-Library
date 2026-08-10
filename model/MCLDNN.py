"""MCLDNN model for automatic modulation classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class MCLDNNConfig(PretrainedConfig):
    """Configuration for :class:`MCLDNNModel`.

    Defaults follow ``scripts/*/MCLDNN.sh``.
    """

    model_type: str = "mcldnn"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        dropout: float = 0.5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.dropout: float = dropout
        self.lstm_hidden: int = lstm_hidden
        self.lstm_layers: int = lstm_layers


class MCLDNNModel(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone."""

    config_class = MCLDNNConfig

    def __init__(self, config: MCLDNNConfig) -> None:
        super().__init__()
        self.config: MCLDNNConfig = config
        self.num_classes: int = config.n_classes
        self.dropout: float = config.dropout

        self.pad_8: nn.ZeroPad2d = nn.ZeroPad2d((7, 0, 0, 0))

        self.conv1: nn.Sequential = nn.Sequential(
            self.pad_8,
            nn.Conv2d(1, 50, kernel_size=(2, 8)),
            nn.ReLU(),
        )
        self.conv2: nn.Sequential = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )
        self.conv3: nn.Sequential = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )
        self.conv4: nn.Sequential = nn.Sequential(
            self.pad_8,
            nn.Conv2d(50, 50, kernel_size=(1, 8)),
            nn.ReLU(),
        )
        self.conv5: nn.Sequential = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding="valid"),
            nn.ReLU(),
        )

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=100,
            hidden_size=config.lstm_hidden,
            batch_first=True,
            num_layers=config.lstm_layers,
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(config.lstm_hidden, config.lstm_hidden),
            nn.SELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, config.lstm_hidden),
            nn.SELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, self.num_classes),
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_2d: torch.Tensor = x_enc.unsqueeze(1)
        x1: torch.Tensor = self.conv1(x_2d)

        x_i: torch.Tensor = x_enc[:, 0:1, :]
        x_q: torch.Tensor = x_enc[:, 1:2, :]
        x2: torch.Tensor = self.conv2(x_i)
        x3: torch.Tensor = self.conv3(x_q)

        x4_input: torch.Tensor = torch.stack([x2, x3], dim=2)
        x4: torch.Tensor = self.conv4(x4_input)

        x1_ext: torch.Tensor = x1.repeat(1, 1, 2, 1)
        x5_input: torch.Tensor = torch.cat([x1_ext, x4], dim=1)
        x5: torch.Tensor = self.conv5(x5_input)

        x: torch.Tensor = x5.squeeze(2)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])
