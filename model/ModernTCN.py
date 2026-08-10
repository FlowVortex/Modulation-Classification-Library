"""ModernTCN model for automatic modulation classification."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class ModernTCNConfig(PretrainedConfig):
    """Configuration for :class:`ModernTCNModel`.

    Defaults follow ``scripts/*/ModernTCN.sh``.
    """

    model_type: str = "moderntcn"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        input_channels: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        d_ff: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        revin: bool = False,
        large_size: Optional[List[int]] = None,
        small_size: Optional[List[int]] = None,
        downsample_ratio: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.input_channels: int = input_channels
        self.patch_len: int = patch_len
        self.stride: int = stride
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.revin: bool = revin
        self.large_size: List[int] = (
            large_size if large_size is not None else [31, 21, 11]
        )
        self.small_size: List[int] = small_size if small_size is not None else [5, 5, 5]
        self.downsample_ratio: int = downsample_ratio


class RevIN(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.affine: bool = affine
        if self.affine:
            self.gamma: nn.Parameter = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta: nn.Parameter = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.stdev + self.mean
        return x


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: Optional[int],
    ) -> None:
        super().__init__()
        self.kernel_size: int = kernel_size
        self.small_kernel: Optional[int] = small_kernel
        padding: int = kernel_size // 2

        self.lkb_origin: nn.Sequential = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if small_kernel is not None:
            self.small_conv: nn.Sequential = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    small_kernel,
                    stride,
                    small_kernel // 2,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(x)
        return out


class Modern_Block(nn.Module):
    def __init__(
        self,
        large_size: int,
        small_size: int,
        dmodel: int,
        dff: int,
        nvars: int,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.dw: ReparamLargeKernelConv = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
        )
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(dmodel)
        self.ffn1: nn.Sequential = nn.Sequential(
            nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=nvars),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=nvars),
            nn.Dropout(drop),
        )
        self.ffn2: nn.Sequential = nn.Sequential(
            nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=dmodel),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=dmodel),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = x
        B: int
        M: int
        D: int
        N: int
        B, M, D, N = x.shape

        x = x.reshape(B, M * D, N)
        x = self.dw(x)

        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M * D, N)

        x = self.ffn1(x)
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2(x)
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)
        return x + res


class Modern_Stage(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        ffn_ratio: int,
        large_size: int,
        small_size: int,
        dmodel: int,
        nvars: int,
        drop: float,
    ) -> None:
        super().__init__()
        d_ffn: int = dmodel * ffn_ratio
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                Modern_Block(large_size, small_size, dmodel, d_ffn, nvars, drop)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCNModel(nn.Module):
    """`ModernTCN <https://openreview.net/forum?id=vpJMJerXHU>`_ backbone.

    Input shape: [Batch, 2, seq_len].
    """

    config_class = ModernTCNConfig

    def __init__(self, config: ModernTCNConfig) -> None:
        super().__init__()
        self.config: ModernTCNConfig = config

        self.seq_len: int = config.seq_len
        self.n_vars: int = config.input_channels
        self.n_classes: int = config.n_classes
        self.patch_size: int = config.patch_len
        self.patch_stride: int = config.stride

        num_stages: int = 3
        stg_num_blocks: List[int] = [config.n_layers] * num_stages
        stg_dims: List[int] = [config.d_model, config.d_model * 2, config.d_model * 4]
        stg_large_size: List[int] = list(config.large_size)
        stg_small_size: List[int] = list(config.small_size)

        ffn_ratio: int = max(1, config.d_ff // config.d_model)
        self.downsample_ratio: int = config.downsample_ratio

        self.revin: Optional[RevIN] = RevIN(self.n_vars) if config.revin else None

        self.downsample_layers: nn.ModuleList = nn.ModuleList()
        stem: nn.Sequential = nn.Sequential(
            nn.Conv1d(
                1, stg_dims[0], kernel_size=self.patch_size, stride=self.patch_stride
            ),
            nn.BatchNorm1d(stg_dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(num_stages - 1):
            down: nn.Sequential = nn.Sequential(
                nn.BatchNorm1d(stg_dims[i]),
                nn.Conv1d(
                    stg_dims[i],
                    stg_dims[i + 1],
                    kernel_size=self.downsample_ratio,
                    stride=self.downsample_ratio,
                ),
            )
            self.downsample_layers.append(down)

        self.stages: nn.ModuleList = nn.ModuleList()
        for i in range(num_stages):
            stage = Modern_Stage(
                num_blocks=stg_num_blocks[i],
                ffn_ratio=ffn_ratio,
                large_size=stg_large_size[i],
                small_size=stg_small_size[i],
                dmodel=stg_dims[i],
                nvars=self.n_vars,
                drop=config.dropout,
            )
            self.stages.append(stage)

        patch_num: int = self.seq_len // self.patch_stride
        final_len: int = patch_num // (self.downsample_ratio ** (num_stages - 1))
        self.head_nf: int = stg_dims[-1] * final_len

        self.class_dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.classifier: nn.Linear = nn.Linear(
            self.n_vars * self.head_nf, self.n_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-2)

        for i in range(len(self.stages)):
            B: int
            M: int
            D: int
            N: int
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len: int = self.patch_size - self.patch_stride
                    pad: torch.Tensor = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)

            x = self.downsample_layers[i](x)
            _, D_new, N_new = x.shape
            x = x.reshape(B, M, D_new, N_new)
            x = self.stages[i](x)

        x = F.gelu(x)
        x = self.class_dropout(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)

    def structural_reparam(self) -> None:
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                pass
