import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    """Bottleneck structure - 1D version"""

    def __init__(
        self, d_model: int, growth_rate: int, bn_size: int, dropout: float
    ) -> None:
        super(_DenseLayer, self).__init__()
        # BN -> ReLU -> Conv1d
        self.layer = nn.Sequential(
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
        self.dropout = float(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        new_features = self.layer(x)
        if self.dropout > 0:
            new_features = F.dropout(
                new_features, p=self.dropout, training=self.training
            )
        # Concatenate along the channel dimension (dimension 1)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """Dense Block consisting of multiple _DenseLayers (1D version)"""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout: float,
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, dropout
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    """
    Transition layer: used for downsampling and channel compression (1D)
    BN + ReLU + 1x1 Conv + 2x2 AvgPooling (1D)
    """

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
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


class Model(nn.Module):
    """`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_ backbone
    Args:
        growth_rate (int): how many filters to add each layer (k)
        block_config (list): how many layers in each dense block
        d_model (int): the number of filters in the first conv layer
        bn_size (int): multiplicative factor for bottleneck layers
        dropout (float): dropout rate after each dense layer
        n_classes (int): number of classes for classification
        reduction (float): compression factor in transition layers
    """

    def __init__(
        self,
        configs,
    ) -> None:
        super(Model, self).__init__()

        self.growth_rate = getattr(configs, "growth_rate", 12)
        self.block_config = getattr(configs, "block_config", (4, 4, 4))
        self.d_model = configs.d_model
        self.bn_size = getattr(configs, "bn_size", 4)
        self.dropout = configs.dropout
        self.n_classes = configs.n_classes
        self.reduction = getattr(configs, "reduction", 0.5)

        # Initial Layer
        self.features = nn.Sequential(
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

        # Build Dense Blocks and Transition Layers
        num_features = self.d_model
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

            # Add Transition layer if it is not the last block
            if i != len(self.block_config) - 1:
                num_output_features = int(num_features * self.reduction)
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_output_features,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_output_features

        # Final Batch Norm
        self.features.add_module("norm5", nn.BatchNorm1d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, self.n_classes)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc: torch.FloatTensor) -> torch.FloatTensor:
        # x_enc shape: (Batch, 2, TimeLength)
        features = self.features(x_enc)

        # Global Average Pooling (1D)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)

        return self.classifier(out)