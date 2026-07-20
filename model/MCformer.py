import torch
from torch import nn


class Model(nn.Module):
    """`MCformer <https://ieeexplore.ieee.org/abstract/document/9685815>`_ backbone
    The input for MCformer is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        n_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(
        self,
        configs,
    ) -> None:
        super(Model, self).__init__()
        
        # 基础参数
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.n_layers = configs.n_layers

        # The rate of dropout layers
        self.dropout = configs.dropout

        # --- Backbone: CNN Embedding ---
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=self.enc_in, out_channels=self.d_model, kernel_size=65, padding="same"
            ),
            nn.ReLU(inplace=True),
        )

        # --- Backbone: Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.n_heads, dim_feedforward=self.d_ff, batch_first=True
        )

        # Stack multiple layers to create the transformer encoder
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # --- Task-Specific Heads ---
        
        # 1. AMC (Automatic Modulation Classification)
        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_amc)
            
        # 2. WTC (Wireless Technology Classification)
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_wtc)
            
        # 3. SS (Spectrum Sensing)
        if self.task_name == 'SS':
            # 采用与原MCformer类似的结构，但输出维度为 n_classes_ss
            self.ss_classifier = self._build_classifier(4 * self.d_model, configs.n_classes_ss)
            
        # 4. AD (Anomaly Detection)
        if self.task_name == 'AD':
            # 异常检测：将 Transformer 输出的每个时间步特征映射回输入维度 (enc_in 通常为 2)
            self.ad_projection = nn.Linear(self.d_model, configs.enc_in)

    def _build_classifier(self, input_dim, num_classes):
        """参考 MCformer 原始设计的分类器结构"""
        return nn.Sequential(
            nn.Linear(input_dim, self.d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_ff, num_classes),
        )

    def feature_extraction(self, x_enc):
        """公共特征提取流程"""
        # 1. CNN Embedding: [B, 2, L] -> [B, d_model, L]
        x_enc = self.embedding(x_enc)
        
        # 2. Prepare for Transformer: [B, d_model, L] -> [B, L, d_model]
        x_enc = torch.transpose(x_enc, 1, 2)

        # 3. Transformer Encoder
        x_dec = self.backbone(x_enc) # [B, L, d_model]
        return x_dec

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        # 取前4个token并展平 (MCformer 论文特性)
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
        # 输出重建结果 [B, L, d_model] -> [B, L, enc_in]
        out = self.ad_projection(feat)
        # 转置回 [B, enc_in, L] 以匹配输入维度
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


if __name__ == "__main__":
    class Configs:
        task_name = 'AMC' # 可选 'AMC', 'WTC', 'SS', 'AD'
        seq_len = 128
        n_classes_amc = 11
        n_classes_wtc = 3
        n_classes_ss = 2
        enc_in = 2        # 异常检测的输入/输出通道数
        d_model = 64
        d_ff = 256
        n_heads = 8
        n_layers = 4
        dropout = 0.1

    print(f"Building MCformer for task: {Configs.task_name}...")
    model = Model(configs=Configs())

    inputs = torch.rand((4, Configs.enc_in, Configs.seq_len))
    print("Input shape:", inputs.shape)
    outputs = model(inputs)

    print("Output shape:", outputs.shape)