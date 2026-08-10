"""CTNet model for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig


class CTNetConfig(PretrainedConfig):
    """Configuration for :class:`CTNetModel`.

    Defaults follow ``scripts/*/CTNet.sh``.
    """

    model_type: str = "ctnet"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.n_layers: int = n_layers
        self.dropout: float = dropout


class CTNetModel(nn.Module):
    """Bidirectional LSTM backbone for modulation classification.

    Args:
        config: Model configuration.
    """

    config_class = CTNetConfig

    def __init__(self, config: CTNetConfig) -> None:
        super().__init__()
        self.config: CTNetConfig = config
        self.d_model: int = config.d_model
        self.n_layers: int = config.n_layers
        self.dropout: float = config.dropout
        self.n_classes: int = config.n_classes

        self.backbone: nn.LSTM = nn.LSTM(
            input_size=2,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )
        self.classifier: nn.Linear = nn.Linear(2 * self.d_model, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length) -> (B, L, C)
        x = torch.transpose(x, 1, 2)
        rnn_out: torch.Tensor
        rnn_out, _ = self.backbone(x)
        last_output: torch.Tensor = rnn_out[:, -1, :]
        return self.classifier(last_output)
