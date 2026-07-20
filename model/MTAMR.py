import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class Model(nn.Module):
    """`MTAMR <https://ieeexplore.ieee.org/document/10471243>`_ 
    An EffectiveMasked Transformer for AMR
    The model processes multimodal sequences (IQ, AP, FT) for modulation recognition.

    Args:
        configs: Configuration object containing model hyperparameters.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len

        # 归一化参数
        self.register_buffer("mean", torch.zeros(1, 7, 1))
        self.register_buffer("std", torch.ones(1, 7, 1))

        self.embedding = nn.Sequential(
            nn.Conv1d(7, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )

        self.pos_encoding = PositionalEncoding(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=configs.n_heads,
            dim_feedforward=2 * self.d_model,
            dropout=configs.dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=configs.n_layers
        )

        # 序列变换模块
        self.W_st1 = nn.Linear(self.d_model, 1)
        self.W_st2 = nn.Linear(self.d_model, self.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_model // 2, configs.n_classes),
        )

        self.mask_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 7),
        )

        # 可学习的损失权重系数
        # 学习 log(sigma^2)，初始化为0对应sigma^2=1
        self.log_sigma2_mp = nn.Parameter(torch.zeros(1))
        self.log_sigma2_ce = nn.Parameter(torch.zeros(1))

    def set_norm_stats(self, mean, std):
        """设置归一化统计量"""
        self.mean = mean.view(1, 7, 1)
        self.std = std.view(1, 7, 1)

    def extract_features(self, x_iq):
        """提取多模态特征"""
        # x_iq: [B, 2, L]
        I = x_iq[:, 0, :]  # [B, L]
        Q = x_iq[:, 1, :]  # [B, L]
        s = torch.complex(I, Q)

        # AP特征
        amp = torch.abs(s)
        phase = torch.angle(s)

        # FT特征
        # s^2 = (I^2 - Q^2) + j(2*I*Q)
        s_squared = torch.complex(I**2 - Q**2, 2 * I * Q)
        # s^4 = (s^2)^2
        s_quartic = s_squared**2

        f1 = torch.log1p(torch.abs(torch.fft.fft(s, dim=-1)))
        f2 = torch.log1p(torch.abs(torch.fft.fft(s_squared, dim=-1)))
        f4 = torch.log1p(torch.abs(torch.fft.fft(s_quartic, dim=-1)))

        # 拼接特征 [B, 7, L]
        x = torch.stack([I, Q, amp, phase, f1, f2, f4], dim=1)

        # 归一化
        return (x - self.mean) / (self.std + 1e-6)

    def apply_random_mask(self, x, mask_ratio):
        """应用随机掩码"""
        B, C, L = x.shape

        if mask_ratio <= 0:
            return x, torch.ones(B, 1, L, device=x.device)

        num_mask = int(L * mask_ratio)
        rand_matrix = torch.rand(B, L, device=x.device)

        # 获取最小的num_mask个值的索引作为掩码位置
        _, mask_indices = torch.topk(rand_matrix, k=num_mask, dim=1, largest=False)

        # 创建掩码矩阵
        mask = torch.ones(B, 1, L, device=x.device, dtype=torch.float32)

        # 使用scatter_设置掩码
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_mask)
        mask.scatter_(2, mask_indices.unsqueeze(1), 0)

        # 扩展掩码到所有通道并应用
        mask_expanded = mask.expand(-1, C, -1)
        masked_x = x * mask_expanded

        return masked_x, mask

    def forward(self, x_iq, mask_ratio=0.0, return_all=False):
        # 提取原始特征
        x_raw = self.extract_features(x_iq)
        B, C, L = x_raw.shape

        # 应用随机掩码
        if self.training and mask_ratio > 0:
            masked_x, mask = self.apply_random_mask(x_raw, mask_ratio)
        else:
            masked_x = x_raw
            mask = torch.ones(B, 1, L, device=x_iq.device)

        # 编码阶段
        enc = self.embedding(masked_x).transpose(1, 2)  # [B, L, d_model]
        enc = self.pos_encoding(enc)
        enc = F.dropout(enc, p=0.1, training=self.training)

        # Transformer编码
        o_n = self.transformer(enc)  # [B, L, d_model]

        # 掩码预测路径
        x_hat = self.mask_predictor(o_n).transpose(1, 2)  # [B, 7, L]

        # 分类路径
        # 序列变换：g(o_n) = softmax(o_n*W1^T) * (o_n*W2)
        attn_scores = self.W_st1(o_n)  # [B, L, 1]
        attn_weights = F.softmax(attn_scores.transpose(1, 2), dim=-1)  # [B, 1, L]
        v = self.W_st2(o_n)  # [B, L, d_model]
        feat = torch.matmul(attn_weights, v).squeeze(1)  # [B, d_model]

        feat = F.dropout(feat, p=0.1, training=self.training)
        logits = self.classifier(feat)

        if return_all:
            return logits, x_hat, x_raw, mask
        else:
            return logits

    def compute_loss(self, logits, x_hat, x_raw, mask, labels):
        B, C, L = x_raw.shape

        # MP Loss
        mask_expanded = mask.expand(-1, C, -1)  # [B, 7, L]

        # 计算每个位置、每个通道的平方误差
        sq_error = (x_hat - x_raw) ** 2  # [B, 7, L]

        masked_positions = 1 - mask_expanded  # [B, 7, L], 1=被遮蔽, 0=保留

        if masked_positions.sum() > 0:
            # 对被遮蔽位置的所有通道求平均
            loss_mp = torch.sum(sq_error * masked_positions) / (
                masked_positions.sum() + 1e-6
            )
        else:
            loss_mp = torch.tensor(0.0, device=logits.device)

        # CE Loss
        loss_ce = F.cross_entropy(logits, labels)

        # 联合损失
        # sigma^2 = exp(log_sigma2)
        sigma2_mp = torch.exp(self.log_sigma2_mp)
        sigma2_ce = torch.exp(self.log_sigma2_ce)
        # 总损失
        loss_total = (
            (1 / (2 * sigma2_mp)) * loss_mp
            + (1 / (2 * sigma2_ce)) * loss_ce
            + torch.log1p(sigma2_mp)
            + torch.log1p(sigma2_ce)
        )

        return loss_total, loss_mp, loss_ce
