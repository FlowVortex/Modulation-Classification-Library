import torch
import torch.nn as nn
import torch.nn.functional as F

class PET(nn.Module):
    def __init__(self, frame_length=128, enc_in=2):
        super(PET, self).__init__()
        self.enc_in = enc_in
        self.frame_length = frame_length
        self.p1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_length * enc_in, 1),
        )

    def forward(self, x):
        # x input: [B, L, enc_in]
        # PET 变换是 I/Q 特定的（sin/cos 旋转），仅当 enc_in == 2 时执行
        if self.enc_in == 2:
            p1_x = self.p1(x)
            sin_x = torch.sin(p1_x)
            cos_x = torch.cos(p1_x)

            x11 = x[:, :, 0] * cos_x
            x12 = x[:, :, 1] * sin_x
            x21 = x[:, :, 0] * sin_x
            x22 = x[:, :, 1] * cos_x

            y1 = x11 + x12
            y2 = x21 - x22
            y1 = torch.unsqueeze(y1, 2)
            y2 = torch.unsqueeze(y2, 2)

            x2 = torch.cat([y1, y2], dim=2)
            x2 = torch.transpose(x2, 1, 2)
            x2 = torch.unsqueeze(x2, 1)
            return x2  # output: [B, 1, 2, L]
        else:
            # enc_in ≠ 2 时，跳过 PET 旋转，直接通过 Linear 变换
            out = self.p1(x)  # [B, 1]
            out = out.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            out = out.expand(-1, -1, self.enc_in, self.frame_length)  # [B, 1, enc_in, L]
            # 加上残差形式的输入变换
            x_flat = x.transpose(1, 2).unsqueeze(1)  # [B, 1, enc_in, L]
            return x_flat + 0.1 * out  # 弱 PET 变换用于非 I/Q 数据


class Model(nn.Module):
    """`PETCGDNN <https://ieeexplore.ieee.org/abstract/document/9507514>`_ backbone
    The input for PETCGDNN is an N*L*2 frame
    Args:
        seq_len (int): the frame length equal to number of sample points
        n_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # --- Backbone: PET + CNN ---
        # PET 输出 [B, 1, enc_in, L]；Conv2d kernel_size[0] 需匹配 enc_in
        self.features = nn.Sequential(
            PET(frame_length=self.seq_len, enc_in=self.enc_in),
            nn.Conv2d(1, 75, kernel_size=(self.enc_in, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 25, kernel_size=(1, 5), padding="valid"),
            nn.ReLU(inplace=True),
        )

        # --- 时序提取 ---
        self.gru = nn.GRU(input_size=25, hidden_size=self.d_model, batch_first=True)

        # --- 任务特定输出头 (Task-Specific Heads) ---
        
        # 1. AMC (Automatic Modulation Classification)
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(self.d_model, configs.n_classes_amc, configs.dropout)
            
        # 2. WTC (Wireless Technology Classification)
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(self.d_model, configs.n_classes_wtc, configs.dropout)
            
        # 3. SS (Spectrum Sensing)
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(self.d_model, 64),
                nn.ReLU(),
                nn.Linear(64, configs.n_classes_ss)
            )
            
        # 4. AD (Anomaly Detection)
        if self.task_name == 'AD':
            self.ad_projection = nn.Linear(self.d_model, configs.enc_in)

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
        公共特征提取流程
        x_enc: [B, L, 2] -> 适配 PET 原始输入
        """
        # 1. PET + CNN 特征提取
        x = self.features(x_enc)               # [B, 25, 1, L_new]
        
        # 2. 维度调整以适应 GRU
        x = torch.squeeze(x, 2)                # [B, 25, L_new]
        x = torch.transpose(x, 1, 2)           # [B, L_new, 25]
        
        # 3. GRU 层
        x, _ = self.gru(x)                     # [B, L_new, d_model]
        return x

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.amc_classifier(feat[:, -1, :])

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.wtc_classifier(feat[:, -1, :])

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.ss_classifier(feat[:, -1, :])

    def ad(self, x_enc):
        feat = self.feature_extraction(x_enc)   # [B, 117, d_model]
        out = self.ad_projection(feat)          # [B, 117, 2]
        out = out.transpose(1, 2)               # [B, 2, 117]
        out = F.interpolate(out, size=self.seq_len, mode='linear', align_corners=False)
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 适配输入：如果输入是 [B, enc_in, L]，需要转为 PET 期望的 [B, L, enc_in]
        if x_enc.shape[1] == self.enc_in and x_enc.shape[2] == self.seq_len:
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