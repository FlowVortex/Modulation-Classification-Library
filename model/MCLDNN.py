import torch
import torch.nn as nn

class Model(nn.Module):
    """
    `MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_
    """
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.num_classes = configs.n_classes
        
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
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding="valid"),
            nn.ReLU(),
        )

        # --- 时序提取 ---
        self.lstm = nn.LSTM(
            input_size=100, 
            hidden_size=128, 
            batch_first=True, 
            num_layers=2
        )

        # --- 分类器 ---
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, x_enc: torch.FloatTensor) -> torch.FloatTensor:
        # x_enc: [B, 2, L]
        
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
        x, _ = self.lstm(x)

        # 6. 分类
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return x