"""MTAMR model for automatic modulation classification."""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class MTAMRConfig(PretrainedConfig):
    """Configuration for :class:`MTAMRModel`.

    Defaults follow ``scripts/*/MTAMR.sh``.
    """

    model_type: str = "mtamr"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe: torch.Tensor = torch.zeros(max_len, d_model)
        position: torch.Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MTAMRModel(nn.Module):
    """`MTAMR <https://ieeexplore.ieee.org/document/10471243>`_ backbone.

    An Effective Masked Transformer for AMR. Processes multimodal sequences
    (IQ, AP, FT) for modulation recognition.
    """

    config_class = MTAMRConfig

    def __init__(self, config: MTAMRConfig) -> None:
        super().__init__()
        self.config: MTAMRConfig = config
        self.d_model: int = config.d_model
        self.seq_len: int = config.seq_len
        self.dropout: float = config.dropout

        self.register_buffer("mean", torch.zeros(1, 7, 1))
        self.register_buffer("std", torch.ones(1, 7, 1))

        self.embedding: nn.Sequential = nn.Sequential(
            nn.Conv1d(7, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.pos_encoding: PositionalEncoding = PositionalEncoding(
            self.d_model, self.seq_len
        )

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        self.W_st1: nn.Linear = nn.Linear(self.d_model, 1)
        self.W_st2: nn.Linear = nn.Linear(self.d_model, self.d_model)

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model // 2, config.n_classes),
        )

        self.mask_predictor: nn.Sequential = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 7),
        )

        self.log_sigma2_mp: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.log_sigma2_ce: nn.Parameter = nn.Parameter(torch.zeros(1))

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set normalization statistics."""
        self.mean = mean.view(1, 7, 1)
        self.std = std.view(1, 7, 1)

    def extract_features(self, x_iq: torch.Tensor) -> torch.Tensor:
        """Extract multimodal features."""
        I: torch.Tensor = x_iq[:, 0, :]
        Q: torch.Tensor = x_iq[:, 1, :]
        s: torch.Tensor = torch.complex(I, Q)

        amp: torch.Tensor = torch.abs(s)
        phase: torch.Tensor = torch.angle(s)

        s_squared: torch.Tensor = torch.complex(I**2 - Q**2, 2 * I * Q)
        s_quartic: torch.Tensor = s_squared**2

        f1: torch.Tensor = torch.log1p(torch.abs(torch.fft.fft(s, dim=-1)))
        f2: torch.Tensor = torch.log1p(torch.abs(torch.fft.fft(s_squared, dim=-1)))
        f4: torch.Tensor = torch.log1p(torch.abs(torch.fft.fft(s_quartic, dim=-1)))

        x: torch.Tensor = torch.stack([I, Q, amp, phase, f1, f2, f4], dim=1)
        return (x - self.mean) / (self.std + 1e-6)

    def apply_random_mask(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking."""
        B: int
        C: int
        L: int
        B, C, L = x.shape

        if mask_ratio <= 0:
            return x, torch.ones(B, 1, L, device=x.device)

        num_mask: int = int(L * mask_ratio)
        rand_matrix: torch.Tensor = torch.rand(B, L, device=x.device)
        _, mask_indices = torch.topk(rand_matrix, k=num_mask, dim=1, largest=False)

        mask: torch.Tensor = torch.ones(B, 1, L, device=x.device, dtype=torch.float32)
        batch_idx: torch.Tensor = (
            torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_mask)
        )
        mask.scatter_(2, mask_indices.unsqueeze(1), 0)

        mask_expanded: torch.Tensor = mask.expand(-1, C, -1)
        masked_x: torch.Tensor = x * mask_expanded
        return masked_x, mask

    def forward(
        self,
        x_iq: torch.Tensor,
        mask_ratio: float = 0.0,
        return_all: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        x_raw: torch.Tensor = self.extract_features(x_iq)
        B: int
        C: int
        L: int
        B, C, L = x_raw.shape

        if self.training and mask_ratio > 0:
            masked_x, mask = self.apply_random_mask(x_raw, mask_ratio)
        else:
            masked_x = x_raw
            mask = torch.ones(B, 1, L, device=x_iq.device)

        enc: torch.Tensor = self.embedding(masked_x).transpose(1, 2)
        enc = self.pos_encoding(enc)
        enc = F.dropout(enc, p=self.dropout, training=self.training)

        o_n: torch.Tensor = self.transformer(enc)
        x_hat: torch.Tensor = self.mask_predictor(o_n).transpose(1, 2)

        attn_scores: torch.Tensor = self.W_st1(o_n)
        attn_weights: torch.Tensor = F.softmax(attn_scores.transpose(1, 2), dim=-1)
        v: torch.Tensor = self.W_st2(o_n)
        feat: torch.Tensor = torch.matmul(attn_weights, v).squeeze(1)

        feat = F.dropout(feat, p=self.dropout, training=self.training)
        logits: torch.Tensor = self.classifier(feat)

        if return_all:
            return logits, x_hat, x_raw, mask
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        x_hat: torch.Tensor,
        x_raw: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B: int
        C: int
        L: int
        B, C, L = x_raw.shape

        mask_expanded: torch.Tensor = mask.expand(-1, C, -1)
        sq_error: torch.Tensor = (x_hat - x_raw) ** 2
        masked_positions: torch.Tensor = 1 - mask_expanded

        if masked_positions.sum() > 0:
            loss_mp: torch.Tensor = torch.sum(sq_error * masked_positions) / (
                masked_positions.sum() + 1e-6
            )
        else:
            loss_mp = torch.tensor(0.0, device=logits.device)

        loss_ce: torch.Tensor = F.cross_entropy(logits, labels)

        sigma2_mp: torch.Tensor = torch.exp(self.log_sigma2_mp)
        sigma2_ce: torch.Tensor = torch.exp(self.log_sigma2_ce)
        loss_total: torch.Tensor = (
            (1 / (2 * sigma2_mp)) * loss_mp
            + (1 / (2 * sigma2_ce)) * loss_ce
            + torch.log1p(sigma2_mp)
            + torch.log1p(sigma2_ce)
        )
        return loss_total, loss_mp, loss_ce
