from typing import List

import torch
from torch import nn

from layers.utils import Activation


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
        """
        The Inception v2 block with 1D convolutions for time series classification.
        """
        super(Inception, self).__init__()

        # The number of convolutional kernels in the inception block
        self.num_kernels = len(kernel_sizes)  # plus one for the max-pooling branch

        if in_channels > 1:
            # If the number of input channels is greater than 1,
            # use a bottleneck layer (1x1 convolution) for dimensionality reduction
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )
        else:
            # If there's only one input channel, skip the bottleneck layer
            # and set bottleneck_channels to 1 for compatibility
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        inception_blocks = [
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
        self.inception_blocks = nn.ModuleList(inception_blocks)

        self.batch_norm = nn.BatchNorm1d(
            num_features=(self.num_kernels + 1) * n_filters
        )
        self.activation = Activation(activation=activation)

    @staticmethod
    def _make_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias: bool = False,
    ) -> nn.Conv1d:
        """
        The helper function to create a 1D convolutional layer.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param padding: Padding size.
        :param stride: Stride size.
        :param bias: Whether to include a bias term.

        :return: (nn.Conv1d) A 1D convolutional layer.
        """
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Inception 1D module.
        The output tensor has shape (batch_size, (num_kernels + 1) * n_filters, seq_len).
        The num_kernels is the number of different kernel sizes used in the inception module.
        Plus one for the max-pooling branch.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, in_channels, seq_len).

        :return: (z_out) (torch.Tensor) Output tensor of shape (batch_size, (num_kernels + 1) * n_filters, seq_len).
        """

        z_bottleneck = self.bottleneck(x)

        z_list = [conv(z_bottleneck) for conv in self.inception_blocks]
        z = torch.cat(z_list, axis=1)

        z_norm = self.batch_norm(z)
        z_out = self.activation(z_norm)

        return z_out


class InceptionFlatten(nn.Module):
    """
    The flattening layer for InceptionTime model.
    """

    def __init__(self, out_features) -> None:
        super(InceptionFlatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class InceptionTime(nn.Module):
    """
    InceptionTime model for time series classification.
    Reference: InceptionTime: Finding AlexNet for Time Series Classification.
    Paper: https://arxiv.org/abs/1909.04939
    Code: http://github.com/hfawaz/InceptionTime

    This model is used as a strong baseline for time series classification tasks.
    In this code we use the InceptionTime as the discriminator in GANs for time series data.
    """

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
        """
        InceptionTime model for time series classification.

        :param seq_len: Length of the input time series.
        :param in_channels: Number of input channels.
        :param n_filters: Number of filters for each convolutional layer.
        :param kernel_sizes: List or tuple of kernel sizes for the convolutional layers.
        :param bottleneck_channels: Number of channels for the bottleneck layer.
        :param n_classes: Number of output classes.
        :param n_blocks: Number of Inception blocks to stack.
        :param activation: Activation function name to use.
        :param use_global_avg_pool: Whether to use global average pooling before the final classification layer.
        :param max_pool_size: The output size of the global average pooling layer.
        :param use_residual: Whether to use residual connections between Inception blocks.
        """
        super(InceptionTime, self).__init__()

        # The number of convolutional kernels in each Inception block
        self.num_kernels = len(kernel_sizes)  # plus one for the max-pooling branch

        # Create the Inception blocks
        self.inception_blocks = nn.ModuleList(
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

        # Whether to use residual connections
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_connections = nn.ModuleList(
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
            self.residual_activations = nn.ModuleList(
                [Activation(activation=activation) for _ in range(n_blocks)]
            )

        else:
            # If not using residual connections, set to None
            self.residual_connections = None
            self.residual_activations = None

        # Whether to use global average pooling before the final classification layer
        self.use_global_avg_pool = use_global_avg_pool

        if use_global_avg_pool:
            # Create the global average pooling layer
            self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=max_pool_size)

            # Create the flattening layer
            self.flatten = InceptionFlatten(
                out_features=(self.num_kernels + 1) * n_filters
            )

            # Create the final classification layer
            self.classifier = nn.Linear(
                in_features=(self.num_kernels + 1) * n_filters, out_features=n_classes
            )
        else:
            # If not using global average pooling, flatten the output of the last Inception block
            self.classifier = nn.Linear(
                in_features=(self.num_kernels + 1) * n_filters * seq_len,
                out_features=n_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the InceptionTime model.

        :param x: (torch.Tensor) Input tensor of shape (batch_size, in_channels, seq_len).

        :return: (torch.Tensor) Output tensor of shape (batch_size, n_classes).
        """
        # (batch_size, in_channels, seq_len)
        out = x

        if self.use_residual:
            # Apply each inception block with its corresponding residual connection and activation
            for inception, res_layer, activation in zip(
                self.inception_blocks,
                self.residual_connections,
                self.residual_activations,
            ):
                z = inception(out)
                res = res_layer(out)
                z = z + res
                out = activation(z)

        else:
            # Apply each inception block sequentially without residual connections
            for inception in self.inception_blocks:
                out = inception(out)

        if self.use_global_avg_pool:
            # Apply global average pooling and flatten the output
            out = self.global_avg_pool(out)
            out = self.flatten(out)
        else:
            # Flatten the output without global average pooling
            out = out.view(out.size(0), -1)  # Flatten without global average pooling

        out = self.classifier(out)

        # Final output shape: (batch_size, n_classes)
        return out


class Model(nn.Module):
    """
    InceptionTime: Finding AlexNet for Time Series Classification <https://arxiv.org/abs/1909.04939>`_ backbone
    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters for each convolutional layer.
        kernel_sizes (list(int)): List or tuple of kernel sizes for the convolutional layers.
        bottleneck_channels (int): Number of channels for the bottleneck layer.
        activation (str): Activation function to use.
    """

    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        
        # 映射 main.py 参数到 InceptionTime 架构
        self.inception_time = InceptionTime(
            seq_len=int(configs.seq_len),
            in_channels=getattr(configs, "input_channels", 2), 
            n_filters=configs.d_model,
            n_blocks=configs.n_layers,
            n_classes=configs.n_classes,
            activation="relu",

            kernel_sizes=getattr(configs, "kernel_sizes", [3, 5, 11]),
            bottleneck_channels=getattr(configs, "bottleneck_channels", 32),
            bias=getattr(configs, "bias", False),
            use_global_avg_pool=getattr(configs, "use_global_avg_pool", True),
            max_pool_size=getattr(configs, "max_pool_size", 1),
            use_residual=getattr(configs, "use_residual", True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 2, Seq_len]
        return self.inception_time(x)
