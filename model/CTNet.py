import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 基础参数引用 (使用与 MCformer 一致的变量名)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.enc_in = configs.enc_in

        # 1. 局部特征提取层 (Convolutional Block)
        # 输入形状: [B, enc_in, L]
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 2. 位置编码 (Position Embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))

        # 3. 全局依赖建模层 (Transformer Block)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_heads, 
            dim_feedforward=self.d_ff, 
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # 4. 任务特定头部 (Task-Specific Heads)
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_amc)
            
        elif self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_wtc)
            
        elif self.task_name == 'SS':
            self.ss_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_ss)
            
        elif self.task_name == 'AD':
            # 异常检测：线性投影回原始维度
            self.ad_projection = nn.Linear(self.d_model, self.enc_in)

    def _build_classifier(self, input_dim, num_classes):
        """参考 MCformer 的分类器结构"""
        return nn.Sequential(
            nn.Linear(input_dim, self.d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_ff, num_classes),
        )

    def feature_extraction(self, x_enc):
        """公共特征提取流程"""
        # x_enc: [B, enc_in, L]
        
        # 1. 卷积提取
        x = self.conv_block(x_enc)  # [B, d_model, L]
        
        # 2. 转换为 Transformer 输入格式 [B, L, d_model]
        x = x.transpose(1, 2)
        
        # 3. 加入位置编码
        x = x + self.pos_embedding
        
        # 4. Transformer Encoder
        x = self.transformer_encoder(x) # [B, L, d_model]
        return x

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 按照 MCformer 逻辑：取前4个token并展平
        x_dec = feat[:, :4, :]
        x_dec = torch.reshape(x_dec, [-1, 4 * self.d_model])
        return self.amc_classifier(x_dec)

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        x_dec = feat[:, :4, :]
        x_dec = torch.reshape(x_dec, [-1, 4 * self.d_model])
        return self.wtc_classifier(x_dec)

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        x_dec = feat[:, :4, :]
        x_dec = torch.reshape(x_dec, [-1, 4 * self.d_model])
        return self.ss_classifier(x_dec)

    def ad(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 重建任务：[B, L, d_model] -> [B, L, enc_in]
        out = self.ad_projection(feat)
        # 转置回 [B, enc_in, L]
        return out.transpose(1, 2)

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