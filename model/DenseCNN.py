"""DenseCNN model for automatic modulation classification."""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class DenseCNNConfig(PretrainedConfig):
    """Configuration for :class:`DenseCNNModel`.

    Defaults follow ``scripts/*/DenseCNN.sh``.
    """

    model_type: str = "densecnn"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 64,
        dropout: float = 0.2,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (4, 4, 4),
        bn_size: int = 4,
        reduction: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.dropout: float = dropout
        self.growth_rate: int = growth_rate
        self.block_config: Tuple[int, ...] = tuple(block_config)
        self.bn_size: int = bn_size
        self.reduction: float = reduction


class _DenseLayer(nn.Module):
    """Bottleneck structure - 1D version."""

    def __init__(
        self, d_model: int, growth_rate: int, bn_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.layer: nn.Sequential = nn.Sequential(
            OrderedDict(
                [
                    ("norm1", nn.BatchNorm1d(d_model)),
                    ("relu1", nn.ReLU(inplace=True)),
                    (
                        "conv1",
                        nn.Conv1d(
                            d_model,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    ("norm2", nn.BatchNorm1d(bn_size * growth_rate)),
                    ("relu2", nn.ReLU(inplace=True)),
                    (
                        "conv2",
                        nn.Conv1d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
        self.dropout: float = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features: torch.Tensor = self.layer(x)
        if self.dropout > 0:
            new_features = F.dropout(
                new_features, p=self.dropout, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """Dense Block consisting of multiple _DenseLayers (1D version)."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout: float,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, dropout
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    """Transition layer for downsampling and channel compression (1D)."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv1d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class DenseCNNModel(nn.Module):
    """`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_ backbone."""

    config_class = DenseCNNConfig

    def __init__(self, config: DenseCNNConfig) -> None:
        super().__init__()
        self.config: DenseCNNConfig = config
        self.growth_rate: int = config.growth_rate
        self.block_config: Tuple[int, ...] = tuple(config.block_config)
        self.d_model: int = config.d_model
        self.bn_size: int = config.bn_size
        self.dropout: float = config.dropout
        self.n_classes: int = config.n_classes
        self.reduction: float = config.reduction

        self.features: nn.Sequential = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv1d(
                            2,
                            self.d_model,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm1d(self.d_model)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features: int = self.d_model
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                dropout=self.dropout,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate

            if i != len(self.block_config) - 1:
                num_output_features: int = int(num_features * self.reduction)
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_output_features,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_output_features

        self.features.add_module("norm5", nn.BatchNorm1d(num_features))
        self.classifier: nn.Linear = nn.Linear(num_features, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.features(x_enc)
        out: torch.Tensor = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        return self.classifier(out)
