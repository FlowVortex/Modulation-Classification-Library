import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Conv-AE: Convolutional Autoencoder for Time Series
    Designed for Multi-task learning (Classification, Sensing, Anomaly Detection)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in  # 输入通道数 (如 I/Q 为 2)
        self.seq_len = configs.seq_len # 输入序列长度
        
        # --- 编码器 (Encoder) ---
        # 逐步减小序列长度，增加通道深度
        self.encoder = nn.Sequential(
            nn.Conv1d(self.enc_in, 32, kernel_size=7, stride=2, padding=3), # L -> L/2
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),          # L/2 -> L/4
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),         # L/4 -> L/8
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # --- 解码器 (Decoder) - 主要用于 AD 任务的重建 ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # L/8 -> L/4
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # L/4 -> L/2
            nn.ReLU(),
            nn.ConvTranspose1d(32, self.enc_in, kernel_size=7, stride=2, padding=3, output_padding=1), # L/2 -> L
        )

        # --- 任务特定输出头 (Task-Specific Heads) ---
        
        # 分类任务 (AMC, WTC) 需要将编码后的特征映射到全连接层
        # 这里使用全局平均池化 (GAP) 处理变长或压缩后的特征
        latent_dim = 128 
        
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(latent_dim, configs.n_classes_amc, configs.dropout)
            
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(latent_dim, configs.n_classes_wtc, configs.dropout)
            
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, configs.n_classes_ss)
            )
            
        # AD 任务直接使用 Decoder 的输出进行重建误差计算

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

    def get_latent_feature(self, x_enc):
        """
        特征提取阶段：
        x_enc: [B, C, L] (Batch, Channels, Length)
        """
        # Encoder 输出: [B, 128, L/8]
        z = self.encoder(x_enc)
        
        # 全局平均池化，提取全局时序特征: [B, 128]
        z_gap = torch.mean(z, dim=-1)
        return z_gap

    def reconstruction(self, x_enc):
        """
        Autoencoder 的重建流程
        """
        z = self.encoder(x_enc)
        x_recon = self.decoder(z)
        # 确保重建后的长度与输入一致 (处理 padding 造成的微小差异)
        if x_recon.shape[-1] != x_enc.shape[-1]:
            x_recon = F.interpolate(x_recon, size=x_enc.shape[-1], mode='linear', align_corners=False)
        return x_recon

    # --- 任务方法 ---

    def amc(self, x_enc):
        feat = self.get_latent_feature(x_enc)
        return self.amc_classifier(feat)

    def wtc(self, x_enc):
        feat = self.get_latent_feature(x_enc)
        return self.wtc_classifier(feat)

    def ss(self, x_enc):
        feat = self.get_latent_feature(x_enc)
        return self.ss_classifier(feat)

    def ad(self, x_enc):
        """
        异常检测：输出重建序列 [B, C, L]
        通常计算 MSE(x_enc, x_recon) 作为 anomaly score
        """
        return self.reconstruction(x_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 注意：某些框架传入 x_enc 为 [B, L, C]，Conv1d 需要 [B, C, L]
        if x_enc.shape[1] == self.seq_len:
            x_enc = x_enc.transpose(1, 2)

        if self.task_name == 'AMC':
            return self.amc(x_enc)
        
        if self.task_name == 'WTC':
            return self.wtc(x_enc)
        
        if self.task_name == 'SS':
            return self.ss(x_enc)
        
        if self.task_name == 'AD':
            return self.ad(x_enc)
            
        return None