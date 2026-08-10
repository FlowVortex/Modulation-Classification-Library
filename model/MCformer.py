"""MCformer model for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig


class MCformerConfig(PretrainedConfig):
    """Configuration for :class:`MCformerModel`.

    Defaults follow ``scripts/*/MCformer.sh``.
    """

    model_type: str = "mcformer"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.dropout: float = dropout


class MCformerModel(nn.Module):
    """`MCformer <https://ieeexplore.ieee.org/abstract/document/9685815>`_ backbone.

    The input for MCformer is a 1*2*L frame ([Batch, 2, seq_len]).
    """

    config_class = MCformerConfig

    def __init__(self, config: MCformerConfig) -> None:
        super().__init__()
        self.config: MCformerConfig = config
        self.seq_len: int = config.seq_len
        self.n_classes: int = config.n_classes
        self.d_model: int = config.d_model
        self.d_ff: int = config.d_ff
        self.n_heads: int = config.n_heads
        self.n_layers: int = config.n_layers
        self.dropout: float = config.dropout

        self.embedding: nn.Sequential = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=self.d_model,
                kernel_size=65,
                padding="same",
            ),
            nn.ReLU(inplace=True),
        )

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            self.d_model,
            self.n_heads,
            dim_feedforward=self.d_ff,
            batch_first=True,
            dropout=self.dropout,
        )
        self.backbone: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_ff, self.n_classes),
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_enc = self.embedding(x_enc)
        x_enc = torch.squeeze(x_enc, dim=2)
        x_enc = torch.transpose(x_enc, 1, 2)

        x_dec: torch.Tensor = self.backbone(x_enc)
        x_dec = x_dec[:, :4, :]
        x_dec = torch.reshape(x_dec, [-1, 4 * self.d_model])
        return self.classifier(x_dec)
