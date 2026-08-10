"""InceptionTime model for automatic modulation classification."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from layers.utils import Activation


class InceptionTimeConfig(PretrainedConfig):
    """Configuration for :class:`InceptionTimeModel`.

    Defaults follow ``scripts/*/InceptionTime.sh``.
    """

    model_type: str = "inceptiontime"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        d_model: int = 32,
        n_layers: int = 6,
        activation: str = "relu",
        input_channels: int = 2,
        kernel_sizes: Optional[List[int]] = None,
        bottleneck_channels: int = 32,
        bias: bool = False,
        use_global_avg_pool: bool = True,
        max_pool_size: int = 1,
        use_residual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.d_model: int = d_model
        self.n_layers: int = n_layers
        self.activation: str = activation
        self.input_channels: int = input_channels
        self.kernel_sizes: List[int] = (
            kernel_sizes if kernel_sizes is not None else [3, 5, 11]
        )
        self.bottleneck_channels: int = bottleneck_channels
        self.bias: bool = bias
        self.use_global_avg_pool: bool = use_global_avg_pool
        self.max_pool_size: int = max_pool_size
        self.use_residual: bool = use_residual


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        activation: str = "relu",
        bias: bool = False,
    ) -> None:
        """Inception v2 block with 1D convolutions for time series classification."""
        super().__init__()
        self.num_kernels: int = len(kernel_sizes)

        if in_channels > 1:
            self.bottleneck: nn.Module = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        inception_blocks: List[nn.Module] = [
            self._make_conv(
                in_channels=bottleneck_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=bias,
            )
            for kernel_size in kernel_sizes
        ] + [
            nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        ]
        self.inception_blocks: nn.ModuleList = nn.ModuleList(inception_blocks)
        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=(self.num_kernels + 1) * n_filters
        )
        self.activation: Activation = Activation(activation=activation)

    @staticmethod
    def _make_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias: bool = False,
    ) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_bottleneck: torch.Tensor = self.bottleneck(x)
        z_list: List[torch.Tensor] = [
            conv(z_bottleneck) for conv in self.inception_blocks
        ]
        z: torch.Tensor = torch.cat(z_list, axis=1)
        z_norm: torch.Tensor = self.batch_norm(z)
        return self.activation(z_norm)


class InceptionFlatten(nn.Module):
    """Flattening layer for InceptionTime."""

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.output_dim: int = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.output_dim)


class InceptionTimeBackbone(nn.Module):
    """InceptionTime backbone for time series classification."""

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        n_filters: int,
        kernel_sizes: List[int] = [3, 5, 11],
        bottleneck_channels: int = 32,
        n_classes: int = 9,
        n_blocks: int = 9,
        activation: str = "relu",
        bias: bool = False,
        use_global_avg_pool: bool = True,
        max_pool_size: int = 1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_kernels: int = len(kernel_sizes)

        self.inception_blocks: nn.ModuleList = nn.ModuleList(
            [
                Inception(
                    in_channels=(
                        in_channels if i == 0 else (self.num_kernels + 1) * n_filters
                    ),
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    bias=bias,
                )
                for i in range(n_blocks)
            ]
        )

        self.use_residual: bool = use_residual
        if self.use_residual:
            self.residual_connections: Optional[nn.ModuleList] = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=(
                                in_channels
                                if i == 0
                                else (self.num_kernels + 1) * n_filters
                            ),
                            out_channels=(self.num_kernels + 1) * n_filters,
                            kernel_size=1,
                            stride=1,
                            bias=bias,
                        ),
                        nn.BatchNorm1d(num_features=(self.num_kernels + 1) * n_filters),
                    )
                    for i in range(n_blocks)
                ]
            )
            self.residual_activations: Optional[nn.ModuleList] = nn.ModuleList(
                [Activation(activation=activation) for _ in range(n_blocks)]
            )
        else:
            self.residual_connections = None
            self.residual_activations = None

        self.use_global_avg_pool: bool = use_global_avg_pool
        if use_global_avg_pool:
            self.global_avg_pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(
                output_size=max_pool_size
            )
            self.flatten: InceptionFlatten = InceptionFlatten(
                out_features=(self.num_kernels + 1) * n_filters
            )
            self.classifier: nn.Linear = nn.Linear(
                in_features=(self.num_kernels + 1) * n_filters, out_features=n_classes
            )
        else:
            self.classifier = nn.Linear(
                in_features=(self.num_kernels + 1) * n_filters * seq_len,
                out_features=n_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x

        if self.use_residual:
            for inception, res_layer, activation in zip(
                self.inception_blocks,
                self.residual_connections,
                self.residual_activations,
            ):
                z: torch.Tensor = inception(out)
                res: torch.Tensor = res_layer(out)
                out = activation(z + res)
        else:
            for inception in self.inception_blocks:
                out = inception(out)

        if self.use_global_avg_pool:
            out = self.global_avg_pool(out)
            out = self.flatten(out)
        else:
            out = out.view(out.size(0), -1)

        return self.classifier(out)


class InceptionTimeModel(nn.Module):
    """InceptionTime: Finding AlexNet for Time Series Classification.

    Paper: https://arxiv.org/abs/1909.04939
    """

    config_class = InceptionTimeConfig

    def __init__(self, config: InceptionTimeConfig) -> None:
        super().__init__()
        self.config: InceptionTimeConfig = config
        self.inception_time: InceptionTimeBackbone = InceptionTimeBackbone(
            seq_len=int(config.seq_len),
            in_channels=config.input_channels,
            n_filters=config.d_model,
            n_blocks=config.n_layers,
            n_classes=config.n_classes,
            activation=config.activation,
            kernel_sizes=list(config.kernel_sizes),
            bottleneck_channels=config.bottleneck_channels,
            bias=config.bias,
            use_global_avg_pool=config.use_global_avg_pool,
            max_pool_size=config.max_pool_size,
            use_residual=config.use_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inception_time(x)
