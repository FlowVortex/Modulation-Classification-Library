import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 软阈值收缩函数 (Garrote Shrinkage)
# ==========================================
class GarroteShrinkage(nn.Module):
    """y=x-\tau^2\ \/x (|x|≥\tau)"""
    def __init__(self, eps=1e-6):
        super(GarroteShrinkage, self).__init__()
        self.eps = eps

    def forward(self, x, tau):
        abs_x = torch.abs(x)
        mask = (abs_x >= tau).float()
        # 防止除以 0
        denominator = x + torch.sign(x + 1e-12) * self.eps
        y = x - (tau**2) / denominator
        return y * mask

# ==========================================
# 2. DP-DRSN 核心残差收缩模块
# ==========================================
class DPDRSNBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(DPDRSNBlock, self).__init__()
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

        # 子网络：学习收缩阈值
        self.subnetwork = nn.Sequential(
            nn.Linear(channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels),
            nn.Sigmoid(),
        )

        self.kappa = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.5))
        self.shrinkage = GarroteShrinkage()

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        # 全局统计量提取 (GAP & GMP)
        abs_out = torch.abs(out)
        alpha = self.subnetwork(torch.mean(abs_out, dim=(2, 3)))
        beta = self.subnetwork(torch.amax(abs_out, dim=(2, 3)))

        # 计算阈值 tau
        gamma_c = torch.clamp(self.gamma, 0, 1)
        tau = self.kappa * (gamma_c * alpha + (1 - gamma_c) * beta)

        # 应用收缩
        out = self.shrinkage(out, tau.view(out.size(0), out.size(1), 1, 1))
        return out + residual

# ==========================================
# 3. 混合特征提取模块 (CNN + LSTM)
# ==========================================
class FeatureExtraction(nn.Module):
    def __init__(self, enc_in=2):
        super(FeatureExtraction, self).__init__()
        # 增加通道数到 32 以提升表达能力
        self.conv_h = nn.Conv2d(1, 32, kernel_size=(3, 1), dilation=2, padding=(2, 0))
        self.conv_v = nn.Conv2d(1, 32, kernel_size=(1, 3), dilation=2, padding=(0, 2))

        # 增加 LSTM 隐藏层到 64，并开启双向；input_size 使用 enc_in
        self.enc_in = enc_in
        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=64, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x shape: [B, enc_in, L]
        # CNN 部分：取前2个通道做空间特征（若 enc_in < 2 则用全部并 padding）
        if x.shape[1] >= 2:
            x_cnn = x[:, :2, :].unsqueeze(1)  # [B, 1, 2, L]
        else:
            x_cnn = x.unsqueeze(1)             # [B, 1, 1, L]

        # 空间特征
        h_cnn = torch.cat([self.conv_h(x_cnn), self.conv_v(x_cnn)], dim=2) # [B, 32, 4, L]

        # 时序特征
        h_lstm_seq, _ = self.lstm(x.transpose(1, 2)) # [B, L, 128]
        # 维度对齐 [B, 128, L] -> [B, 128, 4, L]
        h_lstm = h_lstm_seq.transpose(1, 2).unsqueeze(2).repeat(1, 1, 4, 1)

        # 拼接通道: 32 (CNN) + 128 (LSTM) = 160
        return torch.cat([h_cnn, h_lstm], dim=1)

# ==========================================
# 4. 完整的多任务 DP-DRSN 模型
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = getattr(configs, 'enc_in', 2)
        self.seq_len = getattr(configs, 'seq_len', 128)
        self.dropout = getattr(configs, 'dropout', 0.5)

        # 内部总通道数: 32 (CNN) + 128 (BiLSTM) = 160
        mid_channels = 160 

        # --- 公共 Backbone ---
        self.fe_iq = FeatureExtraction(enc_in=self.enc_in)
        self.fe_ap = FeatureExtraction(enc_in=2)  # A/P 始终是 2 通道

        self.denoiser_iq = nn.Sequential(
            DPDRSNBlock(mid_channels, 9, stride=2),
            DPDRSNBlock(mid_channels, 9, stride=1),
            DPDRSNBlock(mid_channels, 15, stride=2),
            DPDRSNBlock(mid_channels, 15, stride=1),
        )

        self.denoiser_ap = nn.Sequential(
            DPDRSNBlock(mid_channels, 9, stride=2),
            DPDRSNBlock(mid_channels, 9, stride=1),
            DPDRSNBlock(mid_channels, 15, stride=2),
            DPDRSNBlock(mid_channels, 15, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- 任务特定头 (Task Heads) ---
        # 融合后的特征维度: IQ(160) + AP(160) = 320
        input_dim = mid_channels * 2 

        # 1. AMC
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(input_dim, configs.n_classes_amc, self.dropout)
            
        # 2. WTC
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(input_dim, configs.n_classes_wtc, self.dropout)
            
        # 3. SS (二分类/检测)
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, configs.n_classes_ss)
            )
            
        # 4. AD (异常检测/重构)
        if self.task_name == 'AD':
            # 重构输出维度必须是 [Batch, 2 * 128]
            self.ad_projection = nn.Linear(input_dim, self.enc_in * self.seq_len)

    def _build_classifier(self, input_dim, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, num_classes),
        )

    def feature_extraction(self, x_enc):
        """核心特征提取流程"""
        # I/Q 路径
        v_iq_raw = self.fe_iq(x_enc)
        v_iq = self.gap(self.denoiser_iq(v_iq_raw)).view(x_enc.size(0), -1)

        # A/P 路径（仅当 enc_in == 2 即 I/Q 信号时有效）
        if self.enc_in == 2:
            amp = torch.norm(x_enc, p=2, dim=1, keepdim=True)
            phase = torch.atan2(x_enc[:, 1:2, :], x_enc[:, 0:1, :])
            x_ap = torch.cat([amp, phase], dim=1)
            v_ap_raw = self.fe_ap(x_ap)
            v_ap = self.gap(self.denoiser_ap(v_ap_raw)).view(x_enc.size(0), -1)
        else:
            # 当 enc_in ≠ 2 时，复制 I/Q 路径的输出（维度对齐）
            v_ap = v_iq

        # 融合特征 [B, 320]
        return torch.cat([v_iq, v_ap], dim=1)

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.amc_classifier(feat)

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.wtc_classifier(feat)

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.ss_classifier(feat)

    def ad(self, x_enc):
        feat = self.feature_extraction(x_enc)
        out = self.ad_projection(feat)
        # 将 [B, 256] 还原为 [B, 2, 128] 以计算 MSE Loss
        return out.view(x_enc.shape)

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