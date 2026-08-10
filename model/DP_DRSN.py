"""DP-DRSN model for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig


class DP_DRSNConfig(PretrainedConfig):
    """Configuration for :class:`DP_DRSNModel`.

    Defaults follow ``scripts/*/DP_DRSN.sh``.
    """

    model_type: str = "dp_drsn"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes


class GarroteShrinkage(nn.Module):
    """y = x - tau^2 / x  (|x| >= tau)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        abs_x: torch.Tensor = torch.abs(x)
        mask: torch.Tensor = (abs_x >= tau).float()
        denominator: torch.Tensor = x + torch.sign(x + 1e-12) * self.eps
        y: torch.Tensor = x - (tau**2) / denominator
        return y * mask


class DPDRSNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        self.subnetwork: nn.Sequential = nn.Sequential(
            nn.Linear(channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels),
            nn.Sigmoid(),
        )

        self.kappa: nn.Parameter = nn.Parameter(torch.ones(1))
        self.gamma: nn.Parameter = nn.Parameter(torch.full((1,), 0.5))
        self.shrinkage: GarroteShrinkage = GarroteShrinkage()

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.AvgPool2d(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.shortcut(x)
        out: torch.Tensor = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        abs_out: torch.Tensor = torch.abs(out)
        alpha: torch.Tensor = self.subnetwork(torch.mean(abs_out, dim=(2, 3)))
        beta: torch.Tensor = self.subnetwork(torch.amax(abs_out, dim=(2, 3)))

        gamma_c: torch.Tensor = torch.clamp(self.gamma, 0, 1)
        tau: torch.Tensor = self.kappa * (gamma_c * alpha + (1 - gamma_c) * beta)

        out = self.shrinkage(out, tau.view(out.size(0), out.size(1), 1, 1))
        return out + residual


class FeatureExtraction(nn.Module):
    """Hybrid Feature Extraction block combining CNN and LSTM."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_h: nn.Conv2d = nn.Conv2d(
            1, 2, kernel_size=(3, 1), dilation=2, padding=(2, 0)
        )
        self.conv_v: nn.Conv2d = nn.Conv2d(
            1, 2, kernel_size=(1, 3), dilation=2, padding=(0, 2)
        )
        self.lstm: nn.LSTM = nn.LSTM(input_size=2, hidden_size=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cnn: torch.Tensor = x.unsqueeze(1)
        h_cnn: torch.Tensor = torch.cat([self.conv_h(x_cnn), self.conv_v(x_cnn)], dim=2)

        h_lstm_seq: torch.Tensor
        h_lstm_seq, _ = self.lstm(x.transpose(1, 2))
        h_lstm: torch.Tensor = (
            h_lstm_seq.transpose(1, 2).unsqueeze(2).repeat(1, 1, 4, 1)
        )
        return torch.cat([h_cnn, h_lstm], dim=1)


class DP_DRSNModel(nn.Module):
    """`DP-DRSN <https://doi.org/10.3390/ai6080195>`_ backbone.

    The input for DP-DRSN is a 1*2*L frame.
    """

    config_class = DP_DRSNConfig

    def __init__(self, config: DP_DRSNConfig) -> None:
        super().__init__()
        self.config: DP_DRSNConfig = config
        self.n_classes: int = config.n_classes

        self.fe_iq: FeatureExtraction = FeatureExtraction()
        self.fe_ap: FeatureExtraction = FeatureExtraction()

        self.denoiser_iq: nn.Sequential = nn.Sequential(
            DPDRSNBlock(4, 9, stride=2),
            DPDRSNBlock(4, 9, stride=1),
            DPDRSNBlock(4, 15, stride=2),
            DPDRSNBlock(4, 15, stride=1),
        )
        self.denoiser_ap: nn.Sequential = nn.Sequential(
            DPDRSNBlock(4, 9, stride=2),
            DPDRSNBlock(4, 9, stride=1),
            DPDRSNBlock(4, 15, stride=2),
            DPDRSNBlock(4, 15, stride=1),
        )

        self.gap: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.classifier: nn.Linear = nn.Linear(8, self.n_classes)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        amp: torch.Tensor = torch.norm(x_enc, p=2, dim=1, keepdim=True)
        phase: torch.Tensor = torch.atan2(x_enc[:, 1:2, :], x_enc[:, 0:1, :])
        x_ap: torch.Tensor = torch.cat([amp, phase], dim=1)

        v_iq: torch.Tensor = self.gap(self.denoiser_iq(self.fe_iq(x_enc))).view(
            x_enc.size(0), -1
        )
        v_ap: torch.Tensor = self.gap(self.denoiser_ap(self.fe_ap(x_ap))).view(
            x_enc.size(0), -1
        )
        return self.classifier(torch.cat([v_iq, v_ap], dim=1))
