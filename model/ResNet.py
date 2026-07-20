import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    """
    基础 1D 残差块
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 捷径连接 (Shortcut Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Model(nn.Module):
    """
    ResNet for Time Series Classification and Anomaly Detection
    适用于 I/Q 信号 (B, 2, L) 或 单通道序列
    """
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in # 输入维度，如 I/Q 信号为 2
        
        # --- 特征提取网络 (ResNet Backbone) ---
        # 初始层
        self.start_conv = nn.Sequential(
            nn.Conv1d(self.enc_in, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 残差层序列 (类似于 ResNet-18 的 1D 变体)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # 最终特征维度
        self.feature_dim = 512

        # --- 任务特定输出头 ---
        
        # 1. AMC & 2. WTC 分类器
        if self.task_name in ['AMC', 'WTC']:
            n_classes = configs.n_classes_amc if self.task_name == 'AMC' else configs.n_classes_wtc
            self.classifier = self._build_classifier(self.feature_dim, n_classes, configs.dropout)
            
        # 3. SS (Spectrum Sensing)
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, configs.n_classes_ss)
            )
            
        # 4. AD (Anomaly Detection) - 重建或投影头
        if self.task_name == 'AD':
            # 异常检测需要将深层特征映射回原始序列长度
            # 这里采用转置卷积或上采样来恢复序列分辨率
            self.ad_decoder = nn.Sequential(
                nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1), # 恢复 layer4 的下采样
                nn.ReLU(),
                nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # 恢复 layer3 的下采样
                nn.ReLU(),
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 恢复 layer2 的下采样
                nn.ReLU(),
                nn.Conv1d(64, configs.enc_in, kernel_size=3, padding=1)           # 映射回原通道
            )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _build_classifier(self, input_dim, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def feature_extraction(self, x_enc):
        """
        ResNet 特征提取流程
        x_enc: [B, C, L] -> 默认为 [B, 2, L]
        """
        x = self.start_conv(x_enc)    # [B, 64, L]
        x = self.layer1(x)            # [B, 64, L]
        x = self.layer2(x)            # [B, 128, L/2]
        x = self.layer3(x)            # [B, 256, L/4]
        x = self.layer4(x)            # [B, 512, L/8]
        return x

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 全局平均池化 (GAP) 将 [B, 512, L/8] 变为 [B, 512]
        feat_gap = torch.mean(feat, dim=-1)
        return self.classifier(feat_gap)

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat_gap = torch.mean(feat, dim=-1)
        return self.classifier(feat_gap)

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat_gap = torch.mean(feat, dim=-1)
        return self.ss_classifier(feat_gap)

    def ad(self, x_enc):
        """
        AD 任务：输入 [B, 2, L], 输出重建序列 [B, 2, L]
        """
        feat = self.feature_extraction(x_enc) # [B, 512, L/8]
        reconstructed = self.ad_decoder(feat) # [B, 2, L]
        
        # 如果上采样后的长度与输入不一致（由于 stride 取整），进行 F.interpolate
        if reconstructed.shape[-1] != x_enc.shape[-1]:
            reconstructed = F.interpolate(reconstructed, size=x_enc.shape[-1], mode='linear', align_corners=False)
        
        return reconstructed # 形状 [B, 2, L]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 统一输入维度检查，ResNet-1D 期望 [B, C, L]
        if self.task_name == 'AMC':
            return self.amc(x_enc)
        
        if self.task_name == 'WTC':
            return self.wtc(x_enc)
        
        if self.task_name == 'SS':
            return self.ss(x_enc)
        
        if self.task_name == 'AD':
            # 返回转置后的结果以符合某些框架对 [B, L, D] 的预期，或者直接返回 [B, D, L]
            # 这里根据您原代码的 logic: out.transpose(1, 2)
            out = self.ad(x_enc)
            return out
            
        return None