"""PET-CGDNN model for automatic modulation classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class PETCGDNNConfig(PretrainedConfig):
    """Configuration for :class:`PETCGDNNModel`.

    Defaults follow ``scripts/*/PETCGDNN.sh``.
    """

    model_type: str = "petcgdnn"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model


class PET(nn.Module):
    """Phase Estimation Transform block."""

    def __init__(self, frame_length: int = 128) -> None:
        super().__init__()
        self.frame_length: int = frame_length
        self.p1: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_length * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1_x: torch.Tensor = self.p1(x)
        sin_x: torch.Tensor = torch.sin(p1_x)
        cos_x: torch.Tensor = torch.cos(p1_x)

        x11: torch.Tensor = x[:, :, 0] * cos_x
        x12: torch.Tensor = x[:, :, 1] * sin_x
        x21: torch.Tensor = x[:, :, 0] * sin_x
        x22: torch.Tensor = x[:, :, 1] * cos_x

        y1: torch.Tensor = x11 + x12
        y2: torch.Tensor = x21 - x22
        y1 = torch.unsqueeze(y1, 2)
        y2 = torch.unsqueeze(y2, 2)

        x2: torch.Tensor = torch.cat([y1, y2], dim=2)
        x2 = torch.transpose(x2, 1, 2)
        x2 = torch.unsqueeze(x2, 1)
        return x2


class PETCGDNNModel(nn.Module):
    """`PETCGDNN <https://ieeexplore.ieee.org/abstract/document/9507514>`_ backbone.

    The input for PETCGDNN is an N*L*2 frame (internally transposed from [B, 2, L]).

    Args:
        config: Model configuration.
    """

    config_class = PETCGDNNConfig

    def __init__(self, config: PETCGDNNConfig) -> None:
        super().__init__()
        self.config: PETCGDNNConfig = config
        self.n_classes: int = config.n_classes
        self.seq_len: int = config.seq_len
        self.d_model: int = config.d_model

        self.features: nn.Sequential = nn.Sequential(
            PET(frame_length=self.seq_len),
            nn.Conv2d(1, 75, kernel_size=(2, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 25, kernel_size=(1, 5), padding="valid"),
            nn.ReLU(inplace=True),
        )
        self.gru: nn.GRU = nn.GRU(
            input_size=25, hidden_size=self.d_model, batch_first=True
        )
        self.classifier: nn.Linear = nn.Linear(self.d_model, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 2, 1)
        x = self.features(x)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        x = self.classifier(x[:, -1, :])
        return x
