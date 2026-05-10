import torch
from torch import nn


class GarroteShrinkage(nn.Module):
    """y=x-\tau^2\ \/x (|x|≥\tau)"""

    def __init__(self, eps=1e-6):
        super(GarroteShrinkage, self).__init__()
        self.eps = eps

    def forward(self, x, tau):
        abs_x = torch.abs(x)
        mask = (abs_x >= tau).float()
        denominator = x + torch.sign(x + 1e-12) * self.eps
        y = x - (tau**2) / denominator
        return y * mask


class DPDRSNBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(DPDRSNBlock, self).__init__()
        # Convolutional path for feature transformation
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.subnetwork = nn.Sequential(
            nn.Linear(channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels),
            nn.Sigmoid(),
        )

        # Learnable parameters for threshold scaling
        self.kappa = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.5))
        self.shrinkage = GarroteShrinkage()

        # Identity shortcut for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        # Extract global statistics using GAP and GMP
        abs_out = torch.abs(out)
        alpha = self.subnetwork(torch.mean(abs_out, dim=(2, 3)))
        beta = self.subnetwork(torch.amax(abs_out, dim=(2, 3)))

        # Combine statistics with learnable weight gamma (Equation A5)
        gamma_c = torch.clamp(self.gamma, 0, 1)
        tau = self.kappa * (gamma_c * alpha + (1 - gamma_c) * beta)

        # Apply Garrote shrinkage to the feature map
        out = self.shrinkage(out, tau.view(out.size(0), out.size(1), 1, 1))
        return out + residual


class FeatureExtraction(nn.Module):
    """
    Hybrid Feature Extraction block combining CNN and LSTM.
    Extracts spatial and temporal features of I/Q or A/P signals.
    """

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        # CNN path with asymmetric dilated convolutions (dilation=2)
        self.conv_h = nn.Conv2d(1, 2, kernel_size=(3, 1), dilation=2, padding=(2, 0))
        self.conv_v = nn.Conv2d(1, 2, kernel_size=(1, 3), dilation=2, padding=(0, 2))

        # LSTM path for temporal feature dependencies
        self.lstm = nn.LSTM(input_size=2, hidden_size=2, batch_first=True)

    def forward(self, x):
        # Extract spatial features using CNN and concatenate along axis 2
        x_cnn = x.unsqueeze(1)
        h_cnn = torch.cat([self.conv_h(x_cnn), self.conv_v(x_cnn)], dim=2)

        # Extract temporal features and reshape to match CNN spatial dimensions
        h_lstm_seq, _ = self.lstm(x.transpose(1, 2))
        h_lstm = h_lstm_seq.transpose(1, 2).unsqueeze(2).repeat(1, 1, 4, 1)

        # Concatenate spatial and temporal features along the channel axis (axis 1)
        return torch.cat([h_cnn, h_lstm], dim=1)


class Model(nn.Module):
    """`A Lightweight Deep Learning Model for Automatic Modulation Classification Using Dual-Path Deep Residual Shrinkage Network <https://doi.org/10.3390/ai6080195>`_ backbone
    The input for DP-DRSN is a 1*2*L frame
    Args:
        seq_len (int): the frame length equal to number of sample points
        n_classes (int): number of classes for classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_classes = configs.n_classes

        # Create dual-path feature extraction for IQ and AP signals
        self.fe_iq = FeatureExtraction()
        self.fe_ap = FeatureExtraction()

        # Stack 4 DP-DRSN blocks with varying kernels (9 and 15) and strides (2 and 1)
        self.denoiser_iq = nn.Sequential(
            DPDRSNBlock(4, 9, stride=2),
            DPDRSNBlock(4, 9, stride=1),
            DPDRSNBlock(4, 15, stride=2),
            DPDRSNBlock(4, 15, stride=1),
        )

        self.denoiser_ap = nn.Sequential(
            DPDRSNBlock(4, 9, stride=2),
            DPDRSNBlock(4, 9, stride=1),
            DPDRSNBlock(4, 15, stride=2),
            DPDRSNBlock(4, 15, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(8, self.n_classes)

    def forward(self, x_enc: torch.FloatTensor) -> torch.FloatTensor:
        # Internal conversion from I/Q to A/P (Amplitude and Phase)
        amp = torch.norm(x_enc, p=2, dim=1, keepdim=True)
        phase = torch.atan2(x_enc[:, 1:2, :], x_enc[:, 0:1, :])
        x_ap = torch.cat([amp, phase], dim=1)

        # Parallel processing through IQ and AP paths
        v_iq = self.gap(self.denoiser_iq(self.fe_iq(x_enc))).view(x_enc.size(0), -1)
        v_ap = self.gap(self.denoiser_ap(self.fe_ap(x_ap))).view(x_enc.size(0), -1)

        return self.classifier(torch.cat([v_iq, v_ap], dim=1))
