# https://github.com/Hyun-Ryu/emc2net

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Set Transformer 相关组件 (用于分类器) ---


class MAB(nn.Module):
    """Multihead Attention Block"""

    def __init__(self, dim_Q, dim_K, dim_V, n_heads):
        super(MAB, self).__init__()
        self.dim_V = dim_V  # dim_V must be divisible by num_heads
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(dim_Q, n_heads, batch_first=True)

    def forward(self, Q, K):
        # Q: (B, N, D), K: (B, M, D)
        out, _ = self.mha(Q, K, K)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block"""

    def __init__(self, dim_in, dim_out, n_heads, num_inds):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, n_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, n_heads)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multi-head Attention"""

    def __init__(self, d_model, n_heads, num_seeds):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, d_model))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d_model, d_model, d_model, n_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# --- EMC2-Net 主模型 ---


class ResidualBlock(nn.Module):
    """等化器中的 Conv1D 残差块"""

    def __init__(self, channels=2, kernel_size=65):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class Model(nn.Module):
    """`EMC2-Net <https://ieeexplore.ieee.org/abstract/document/10096687>`_ backbone
    Equalized Matching-filtering and Constellation-Consistency Network.
    The input for EMC2-Net is a 2*L frame (represented as [Batch, 2, seq_len]).

    Args:
        seq_len (int): the frame length equal to number of sample points (L).
        decimation_factor (int): the factor used for downsampling/decimation 
            (e.g., 8 to convert 8192 samples to 1024 symbols).
        n_classes (int): number of classes for classification.
    """
    def __init__(
        self,
        configs,
    ) -> None:
        super(Model, self).__init__()

        self.n_classes = configs.n_classes
        self.seq_len = configs.seq_len

        # 1. Equalizer (均衡器部分)
        self.equalizer = nn.Sequential(
            ResidualBlock(channels=2, kernel_size=65),
            ResidualBlock(channels=2, kernel_size=65),
        )
        self.decimation_factor = getattr(configs, "decimation_factor", 1)

        # 2. Classifier (分类器部分: Set Transformer)
        # 输入维度是 2 (I/Q), 映射到 128 维
        self.sig2con_fc = nn.Linear(2, 128)
        self.isab1 = ISAB(dim_in=128, dim_out=128, n_heads=4, num_inds=64)
        self.isab2 = ISAB(dim_in=128, dim_out=128, n_heads=4, num_inds=64)
        self.pma = PMA(d_model=128, n_heads=4, num_seeds=1)

        self.dropout = nn.Dropout(configs.dropout)
        self.fc_out = nn.Linear(128, configs.n_classes)

    def forward(self, x):
        """
        Input shape: (Batch, 2, 8192) -> I/Q sequential signals
        """
        # --- 等化阶段 ---
        x = self.equalizer(x)

        # Zero-mean
        x = x - torch.mean(x, dim=2, keepdim=True)

        # Decimation (模拟匹配滤波后的抽取过程)
        # 论文提到 N=8, 从 8192 降采样到 1024 个符号
        x = x[:, :, :: self.decimation_factor]

        # UPNorm (Unit Power Normalization)
        power = torch.mean(x**2, dim=[1, 2], keepdim=True)
        x = x / torch.sqrt(power + 1e-8)

        # --- 星座点转换 (Sig2Con) ---
        # 转置为 (Batch, 1024, 2) 视为点集
        x = x.transpose(1, 2)

        # --- 分类阶段 ---
        x = self.sig2con_fc(x)
        x = self.isab1(x)
        x = self.isab2(x)
        x = self.pma(x)  # (Batch, 1, 128)

        x = x.squeeze(1)
        x = self.dropout(x)
        logits = self.fc_out(x)

        return logits


