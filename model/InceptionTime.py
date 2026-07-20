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
    def _make_conv(in_channels, out_channels, kernel_size, padding, stride, bias=False):
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_bottleneck = self.bottleneck(x)
        z_list = [conv(z_bottleneck) for conv in self.inception_blocks]
        z = torch.cat(z_list, axis=1)
        z_norm = self.batch_norm(z)
        z_out = self.activation(z_norm)
        return z_out

class InceptionFlatten(nn.Module):
    def __init__(self, out_features) -> None:
        super(InceptionFlatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: List[int] = [3, 5, 11],
        bottleneck_channels: int = 32,
        n_blocks: int = 9,
        activation: str = "relu",
        bias: bool = False,
        use_residual: bool = True,
    ) -> None:
        super(InceptionTime, self).__init__()
        self.num_kernels = len(kernel_sizes)
        self.inception_blocks = nn.ModuleList([
            Inception(
                in_channels=(in_channels if i == 0 else (self.num_kernels + 1) * n_filters),
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                activation=activation,
                bias=bias,
            ) for i in range(n_blocks)
        ])

        self.use_residual = use_residual
        if self.use_residual:
            self.residual_connections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=(in_channels if i == 0 else (self.num_kernels + 1) * n_filters),
                        out_channels=(self.num_kernels + 1) * n_filters,
                        kernel_size=1, stride=1, bias=bias,
                    ),
                    nn.BatchNorm1d(num_features=(self.num_kernels + 1) * n_filters),
                ) for i in range(n_blocks)
            ])
            self.residual_activations = nn.ModuleList([Activation(activation=activation) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.use_residual:
            for inception, res_layer, activation in zip(self.inception_blocks, self.residual_connections, self.residual_activations):
                z = inception(out)
                res = res_layer(out)
                out = activation(z + res)
        else:
            for inception in self.inception_blocks:
                out = inception(out)
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
        self.task_name = configs.task_name
        
        # 基础特征提取能力 (Backbone)
        self.backbone = InceptionTime(
            in_channels=getattr(configs, "enc_in", 2), 
            n_filters=configs.d_model,
            n_blocks=configs.n_layers,
            kernel_sizes=getattr(configs, "kernel_sizes", [3, 5, 11]),
            bottleneck_channels=getattr(configs, "bottleneck_channels", 32),
            activation="relu",
            bias=getattr(configs, "bias", False),
            use_residual=getattr(configs, "use_residual", True),
        )

        # 特征维度计算
        self.feature_dim = (len(getattr(configs, "kernel_sizes", [3, 5, 11])) + 1) * configs.d_model
        
        # 池化与展平逻辑 (用于分类任务)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=getattr(configs, "max_pool_size", 1))
        self.flatten = InceptionFlatten(out_features=self.feature_dim)

        # 任务特定输出头 (Task-Specific Heads)
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(self.feature_dim, configs.n_classes_amc, configs.dropout)
            
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(self.feature_dim, configs.n_classes_wtc, configs.dropout)
            
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, configs.n_classes_ss)
            )
            
        if self.task_name == 'AD':
            # 映射回输入维度进行重建
            self.ad_projection = nn.Linear(self.feature_dim, configs.enc_in)

    def _build_classifier(self, input_dim, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def feature_extraction(self, x_enc):
        """
        x_enc: [B, C, L] -> 经过 Backbone 处理
        """
        # [B, feature_dim, L]
        feat = self.backbone(x_enc)
        return feat

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # InceptionTime 通常使用全局池化处理变长序列
        feat = self.global_avg_pool(feat)
        feat = self.flatten(feat)
        return self.amc_classifier(feat)

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat = self.global_avg_pool(feat)
        feat = self.flatten(feat)
        return self.wtc_classifier(feat)

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat = self.global_avg_pool(feat)
        feat = self.flatten(feat)
        return self.ss_classifier(feat)

    def ad(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # AD任务通常输出序列重建: [B, L, D]
        feat = feat.transpose(1, 2) # [B, L, feature_dim]
        out = self.ad_projection(feat) # [B, L, enc_in]
        return out.transpose(1, 2) # [B, enc_in, L]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'AMC':
            return self.amc(x_enc)
        
        if self.task_name == 'WTC':
            return self.wtc(x_enc)
        
        if self.task_name == 'SS':
            return self.ss(x_enc)
        
        if self.task_name == 'AD':
            return self.ad(x_enc)
            
        return None