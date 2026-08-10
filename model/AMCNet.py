"""AMCNet model for automatic modulation classification."""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class AMCNetConfig(PretrainedConfig):
    """Configuration for :class:`AMCNetModel`.

    Defaults follow ``scripts/*/AMCNet.sh``.
    """

    model_type: str = "amcnet"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        dropout: float = 0.5,
        conv_chan_list: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        # MultiScaleModule emits (d_model // 3) * 3 channels; stem must match.
        msm_channels: int = (d_model // 3) * 3
        self.conv_chan_list: List[int] = (
            conv_chan_list
            if conv_chan_list is not None
            else [msm_channels, 64, 128, 256]
        )


class Conv_Block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.in_c: int = in_channel
        self.out_c: int = out_channel

        self.conv_block: nn.Sequential = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batchsize, C, H, W]"""
        return self.conv_block(x)


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel: int = 128) -> None:
        super().__init__()
        self.out_c: int = out_channel

        self.conv_3: nn.Sequential = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_5: nn.Sequential = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_7: nn.Sequential = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        y1: torch.Tensor = self.conv_3(x_enc)
        y2: torch.Tensor = self.conv_5(x_enc)
        y3: torch.Tensor = self.conv_7(x_enc)
        return torch.cat([y1, y2, y3], dim=1)


class TinyMLP(nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.N: int = N
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AdaCorrModule(nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.Im: TinyMLP = TinyMLP(N)
        self.Re: TinyMLP = TinyMLP(N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init: torch.Tensor = copy.deepcopy(x)
        x = torch.fft.fft(x, dim=-1)
        X_re: torch.Tensor = torch.real(x)
        X_im: torch.Tensor = torch.imag(x)
        h_re: torch.Tensor = self.Re(X_re)
        h_im: torch.Tensor = self.Im(X_im)
        x = torch.mul(h_re, X_re) + 1j * torch.mul(h_im, X_im)
        x = torch.real(torch.fft.ifft(x, dim=-1))
        return x + x_init


class FeaFusionModule(nn.Module):
    def __init__(
        self, num_attention_heads: int, input_size: int, hidden_size: int
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads: int = num_attention_heads
        self.attention_head_size: int = int(hidden_size / num_attention_heads)
        self.all_head_size: int = hidden_size

        self.key_layer: nn.Linear = nn.Linear(input_size, hidden_size)
        self.query_layer: nn.Linear = nn.Linear(input_size, hidden_size)
        self.value_layer: nn.Linear = nn.Linear(input_size, hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_size = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key: torch.Tensor = self.key_layer(x)
        query: torch.Tensor = self.query_layer(x)
        value: torch.Tensor = self.value_layer(x)

        key_heads: torch.Tensor = self.trans_to_multiple_heads(key)
        query_heads: torch.Tensor = self.trans_to_multiple_heads(query)
        value_heads: torch.Tensor = self.trans_to_multiple_heads(value)

        attention_scores: torch.Tensor = torch.matmul(
            query_heads, key_heads.permute(0, 1, 3, 2)
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs: torch.Tensor = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context: torch.Tensor = torch.matmul(attention_probs, value_heads)
        shape = context.size()
        context = context.contiguous().view(shape[0], -1, shape[-1])
        return context


class AMCNetModel(nn.Module):
    """`AMCNet <https://ieeexplore.ieee.org/document/10097070>`_ backbone.

    The input for AMCNet is a 2*L frame (represented as [Batch, 2, seq_len]).
    """

    config_class = AMCNetConfig

    def __init__(self, config: AMCNetConfig) -> None:
        super().__init__()
        self.config: AMCNetConfig = config

        self.sig_len: int = config.seq_len
        self.extend_channel: int = config.d_model
        self.latent_dim: int = config.d_ff
        self.n_classes: int = config.n_classes
        self.num_heads: int = config.n_heads
        self.conv_chan_list: List[int] = list(config.conv_chan_list)
        self.stem_layers_num: int = len(self.conv_chan_list) - 1

        self.ACM: AdaCorrModule = AdaCorrModule(self.sig_len)
        self.MSM: MultiScaleModule = MultiScaleModule(self.extend_channel)
        self.FFM: FeaFusionModule = FeaFusionModule(
            self.num_heads, self.sig_len, self.sig_len
        )

        self.Conv_stem: nn.Sequential = nn.Sequential()
        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(
                f"conv_stem_{t}",
                Conv_Block(self.conv_chan_list[t], self.conv_chan_list[t + 1]),
            )

        self.GAP: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(config.dropout),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.ACM(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM(x.squeeze(2))
        x = self.GAP(x)
        y: torch.Tensor = self.classifier(x.squeeze(2))
        return y
