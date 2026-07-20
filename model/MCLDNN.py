import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    `MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_
    """
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.task_name = configs.task_name
        
        # --- 公共特征提取网络 (Backbone) ---
        self.pad_8 = nn.ZeroPad2d((7, 0, 0, 0)) 

        # --- 路径 1: Combined (2D) ---
        self.conv1 = nn.Sequential(
            self.pad_8,
            nn.Conv2d(1, 50, kernel_size=(2, 8)),
            nn.ReLU(),
        )

        # --- 路径 2: Individual (1D) ---
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            self.pad_8,
            nn.Conv2d(50, 50, kernel_size=(1, 8)),
            nn.ReLU(),
        )

        # --- 融合层 (Conv5) ---
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding=(0, 2)),
            nn.ReLU(),
        )

        # --- 时序提取 ---
        self.lstm = nn.LSTM(
            input_size=100, 
            hidden_size=128, 
            batch_first=True, 
            num_layers=2
        )

        # --- 任务特定输出头 (Task-Specific Heads) ---
        
        # 1. AMC (Automatic Modulation Classification)
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(128, configs.n_classes_amc, configs.dropout)
            
        # 2. WTC (Wireless Technology Classification)
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(128, configs.n_classes_wtc, configs.dropout)
            
        # 3. SS (Spectrum Sensing) - 通常是二分类或信号存在性检测
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, configs.n_classes_ss) # 通常 n_classes_ss=2
            )
            
        # 4. AD (Anomaly Detection) - 参考 Autoformer，通过重建或投影特征来检测
        if self.task_name == 'AD':
            # 异常检测可能需要映射回原始维度空间或映射到一个得分空间
            self.ad_projection = nn.Linear(128, configs.enc_in) # 映射回输入维度进行重建

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
        x_enc: [B, 2, L]
        """
        # 1. 2D 分支
        x_2d = x_enc.unsqueeze(1)            # [B, 1, 2, L]
        x1 = self.conv1(x_2d)                # [B, 50, 1, L] 

        # 2. I/Q 分立分支
        x_i = x_enc[:, 0:1, :]               # [B, 1, L]
        x_q = x_enc[:, 1:2, :]               # [B, 1, L]
        x2 = self.conv2(x_i)                 # [B, 50, L]
        x3 = self.conv3(x_q)                 # [B, 50, L]

        # 3. 拼接 I/Q 路进入 Conv4
        x4_input = torch.stack([x2, x3], dim=2) # [B, 50, 2, L]
        x4 = self.conv4(x4_input)               # [B, 50, 2, L]

        # 4. 最终融合 (Concatenate 2)
        # 对齐高度：将 x1 从 [B, 50, 1, L] 变为 [B, 50, 2, L]
        x1_ext = x1.repeat(1, 1, 2, 1)          
        x5_input = torch.cat([x1_ext, x4], dim=1) # [B, 100, 2, L]
        x5 = self.conv5(x5_input)                 # [B, 100, 1, L-4]

        # 5. LSTM 层
        x = x5.squeeze(2)                       # [B, 100, L-4]
        x = x.transpose(1, 2)                   # [B, L-4, 100]
        x, _ = self.lstm(x)                     # [B, L-4, 128]
        self.lstm.flatten_parameters() 
        return x

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 取最后一个时隙的输出进行分类
        return self.amc_classifier(feat[:, -1, :])

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        return self.wtc_classifier(feat[:, -1, :])

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 频谱感知可以基于全局特征
        return self.ss_classifier(feat[:, -1, :])

    def ad(self, x_enc):
        L_ori = x_enc.shape[2]  # 记录原始序列长度
        feat = self.feature_extraction(x_enc)  # [B, L-4, 128]
        # 异常检测：输出每个时间步的投影值，用于计算重建误差
        out = self.ad_projection(feat)         # [B, L-4, enc_in]
        out = out.transpose(1, 2)              # [B, enc_in, L-4]
        # 插值恢复原始长度
        out = F.interpolate(out, size=L_ori, mode='linear', align_corners=False)
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'AMC':
            return self.amc(x_enc)
        
        if self.task_name == 'WTC':
            return self.wtc(x_enc)
        
        if self.task_name == 'SS':
            return self.ss(x_enc)
        
        if self.task_name == 'AD':
            # 注意：AD任务输出通常是整个序列的重建 [B, L, D]
            return self.ad(x_enc)
            
        return None