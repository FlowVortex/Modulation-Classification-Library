"""EMC2-Net model for automatic modulation classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class EMC2NetConfig(PretrainedConfig):
    """Configuration for :class:`EMC2NetModel`.

    Defaults follow ``scripts/*/EMC2Net.sh``.
    """

    model_type: str = "emc2net"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.5,
        decimation_factor: int = 1,
        num_inds: int = 64,
        num_seeds: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.decimation_factor: int = decimation_factor
        self.num_inds: int = num_inds
        self.num_seeds: int = num_seeds


class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, n_heads: int) -> None:
        super().__init__()
        self.dim_V: int = dim_V
        self.n_heads: int = n_heads
        self.mha: nn.MultiheadAttention = nn.MultiheadAttention(
            dim_Q, n_heads, batch_first=True
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        out, _ = self.mha(Q, K, K)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block."""

    def __init__(self, dim_in: int, dim_out: int, n_heads: int, num_inds: int) -> None:
        super().__init__()
        self.I: nn.Parameter = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0: MAB = MAB(dim_out, dim_in, dim_out, n_heads)
        self.mab1: MAB = MAB(dim_in, dim_out, dim_out, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H: torch.Tensor = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multi-head Attention."""

    def __init__(self, d_model: int, n_heads: int, num_seeds: int) -> None:
        super().__init__()
        self.S: nn.Parameter = nn.Parameter(torch.Tensor(1, num_seeds, d_model))
        nn.init.xavier_uniform_(self.S)
        self.mab: MAB = MAB(d_model, d_model, d_model, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ResidualBlock(nn.Module):
    """Equalizer Conv1D residual block."""

    def __init__(self, channels: int = 2, kernel_size: int = 65) -> None:
        super().__init__()
        padding: int = (kernel_size - 1) // 2
        self.conv1: nn.Conv1d = nn.Conv1d(
            channels, channels, kernel_size, padding=padding
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            channels, channels, kernel_size, padding=padding
        )
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        out: torch.Tensor = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class EMC2NetModel(nn.Module):
    """`EMC2-Net <https://ieeexplore.ieee.org/abstract/document/10096687>`_ backbone.

    Equalized Matching-filtering and Constellation-Consistency Network.
    Input shape: [Batch, 2, seq_len].
    """

    config_class = EMC2NetConfig

    def __init__(self, config: EMC2NetConfig) -> None:
        super().__init__()
        self.config: EMC2NetConfig = config
        self.n_classes: int = config.n_classes
        self.seq_len: int = config.seq_len
        self.d_model: int = config.d_model
        self.decimation_factor: int = config.decimation_factor

        self.equalizer: nn.Sequential = nn.Sequential(
            ResidualBlock(channels=2, kernel_size=65),
            ResidualBlock(channels=2, kernel_size=65),
        )

        self.sig2con_fc: nn.Linear = nn.Linear(2, self.d_model)
        self.isab1: ISAB = ISAB(
            dim_in=self.d_model,
            dim_out=self.d_model,
            n_heads=config.n_heads,
            num_inds=config.num_inds,
        )
        self.isab2: ISAB = ISAB(
            dim_in=self.d_model,
            dim_out=self.d_model,
            n_heads=config.n_heads,
            num_inds=config.num_inds,
        )
        self.pma: PMA = PMA(
            d_model=self.d_model, n_heads=config.n_heads, num_seeds=config.num_seeds
        )

        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.fc_out: nn.Linear = nn.Linear(self.d_model, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.equalizer(x)
        x = x - torch.mean(x, dim=2, keepdim=True)
        x = x[:, :, :: self.decimation_factor]

        power: torch.Tensor = torch.mean(x**2, dim=[1, 2], keepdim=True)
        x = x / torch.sqrt(power + 1e-8)

        x = x.transpose(1, 2)
        x = self.sig2con_fc(x)
        x = self.isab1(x)
        x = self.isab2(x)
        x = self.pma(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.fc_out(x)
