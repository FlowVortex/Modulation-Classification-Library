"""CDAT model for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class CDATConfig(PretrainedConfig):
    """Configuration for :class:`CDATModel`.

    Defaults follow ``scripts/*/CDAT.sh``.
    """

    model_type: str = "cdat"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.dropout: float = dropout


class DSConv(nn.Module):
    """Depth-wise Separable Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.depthwise: nn.Conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise: nn.Conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CDA(nn.Module):
    """Convolutional Dual-Attention."""

    def __init__(self, d_model: int, n_heads: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_head: int = d_model // n_heads

        self.q_proj: DSConv = DSConv(
            d_model, d_model, kernel_size, padding=kernel_size // 2
        )
        self.k_proj: DSConv = DSConv(
            d_model, d_model, kernel_size, padding=kernel_size // 2
        )
        self.v_proj: DSConv = DSConv(
            d_model, d_model, kernel_size, padding=kernel_size // 2
        )

        self.ac_convs: nn.ModuleList = nn.ModuleList(
            [
                DSConv(self.d_head, self.d_head, kernel_size, padding=kernel_size // 2)
                for _ in range(n_heads)
            ]
        )

        self.ma_conv: DSConv = DSConv(
            d_model, d_model, kernel_size, padding=kernel_size // 2
        )
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, L]
        B: int
        C: int
        L: int
        B, C, L = x.shape

        Q: torch.Tensor = self.q_proj(x)
        K: torch.Tensor = self.k_proj(x)
        V: torch.Tensor = self.v_proj(x)

        head_outputs: list[torch.Tensor] = []
        for i in range(self.n_heads):
            start: int = i * self.d_head
            end: int = (i + 1) * self.d_head
            qi: torch.Tensor = Q[:, start:end, :]
            ki: torch.Tensor = K[:, start:end, :]
            vi: torch.Tensor = V[:, start:end, :]

            qi_t: torch.Tensor = qi.transpose(1, 2)
            attn_weight: torch.Tensor = torch.matmul(qi_t, ki) * (self.d_head**-0.5)
            attn_weight = F.softmax(attn_weight, dim=-1)
            apos: torch.Tensor = torch.matmul(attn_weight, vi.transpose(1, 2))
            apos = apos.transpose(1, 2)

            ac_weight: torch.Tensor = self.sigmoid(self.ac_convs[i](qi))
            ac: torch.Tensor = ac_weight * vi
            head_outputs.append(apos + ac)

        ma: torch.Tensor = torch.cat(head_outputs, dim=1)
        return self.ma_conv(ma)


class CDATBlock(nn.Module):
    """CDAT Transformer block."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ln1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.attn: CDA = CDA(d_model, n_heads)
        self.ln2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        x = x.transpose(1, 2)
        x = self.ln1(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(self.attn(x))

        residual = x
        x = x.transpose(1, 2)
        x = self.ln2(x)
        x = self.ffn(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        return x


class CDATModel(nn.Module):
    """`CDAT <https://link.springer.com/article/10.1007/s10489-024-06202-6>`_ backbone.

    The input for CDAT is a 2*L frame (represented as [Batch, 2, seq_len]).
    """

    config_class = CDATConfig

    def __init__(self, config: CDATConfig) -> None:
        super().__init__()
        self.config: CDATConfig = config
        c: int = config.d_model

        self.stage1_embed: nn.Conv1d = nn.Conv1d(
            2, c, kernel_size=7, stride=2, padding=3
        )
        self.stage1_block: CDATBlock = CDATBlock(
            c, config.n_heads, config.d_ff, config.dropout
        )

        self.stage2_embed: nn.Conv1d = nn.Conv1d(
            c, c * 2, kernel_size=5, stride=2, padding=2
        )
        self.stage2_block: CDATBlock = CDATBlock(
            c * 2, config.n_heads, config.d_ff, config.dropout
        )

        self.stage3_embed: nn.Conv1d = nn.Conv1d(
            c * 2, c * 4, kernel_size=3, stride=2, padding=1
        )
        self.stage3_block: CDATBlock = CDATBlock(
            c * 4, config.n_heads, config.d_ff, config.dropout
        )

        self.stage4_embed: nn.Conv1d = nn.Conv1d(
            c * 4, c * 8, kernel_size=3, stride=2, padding=1
        )
        self.stage4_block: CDATBlock = CDATBlock(
            c * 8, config.n_heads, config.d_ff, config.dropout
        )

        self.pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier: nn.Linear = nn.Linear(c * 8, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1_embed(x)
        x = self.stage1_block(x)
        x = self.stage2_embed(x)
        x = self.stage2_block(x)
        x = self.stage3_embed(x)
        x = self.stage3_block(x)
        x = self.stage4_embed(x)
        x = self.stage4_block(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
