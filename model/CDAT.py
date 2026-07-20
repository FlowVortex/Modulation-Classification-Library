import torch
from torch import nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """
    Depth-wise Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DSConv, self).__init__()
        # Depthwise: groups=in_channels
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        # Pointwise
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CDA(nn.Module):
    """
    Convolutional Dual-Attention
    """

    def __init__(self, d_model, n_heads, kernel_size=3):
        super(CDA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V
        self.q_proj = DSConv(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.k_proj = DSConv(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.v_proj = DSConv(d_model, d_model, kernel_size, padding=kernel_size // 2)

        # DSConv(Q)
        self.ac_convs = nn.ModuleList(
            [
                DSConv(self.d_head, self.d_head, kernel_size, padding=kernel_size // 2)
                for _ in range(n_heads)
            ]
        )

        self.ma_conv = DSConv(d_model, d_model, kernel_size, padding=kernel_size // 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, L]
        B, C, L = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        head_outputs = []
        for i in range(self.n_heads):
            # 分割多头
            start, end = i * self.d_head, (i + 1) * self.d_head
            qi, ki, vi = Q[:, start:end, :], K[:, start:end, :], V[:, start:end, :]

            # Apos
            # [B, d_head, L] -> [B, L, d_head]
            qi_t = qi.transpose(1, 2)
            attn_weight = torch.matmul(qi_t, ki) * (self.d_head**-0.5)
            attn_weight = F.softmax(attn_weight, dim=-1)
            apos = torch.matmul(attn_weight, vi.transpose(1, 2))
            apos = apos.transpose(1, 2)

            # Ac
            ac_weight = self.sigmoid(self.ac_convs[i](qi))
            ac = ac_weight * vi

            head_outputs.append(apos + ac)

        ma = torch.cat(head_outputs, dim=1)
        out = self.ma_conv(ma)
        return out


class CDATBlock(nn.Module):
    """
    CDAT Transformer 块
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(CDATBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CDA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.ln1(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(self.attn(x))

        residual = x
        x = x.transpose(1, 2)
        x = self.ln2(x)
        x = self.ffn(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        return x

class Model(nn.Module):
    """`CDAT <https://link.springer.com/article/10.1007/s10489-024-06202-6>`_ backbone
    The input for CDAT is a 2*L frame (represented as [Batch, 2, seq_len])

    Args:
        configs: A configuration object containing:
            d_model (int): the initial embedding dimension (base channel size).
            n_heads (int): number of heads in the Convolutional Dual-Attention.
            d_ff (int): the hidden dimension of the feedforward network in CDAT blocks.
            n_classes (int): number of classes for classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in

        # Backbone (Common Feature Extractor)
        c = configs.d_model
        self.stage1_embed = nn.Conv1d(self.enc_in, c, kernel_size=7, stride=2, padding=3)
        self.stage1_block = CDATBlock(c, configs.n_heads, configs.d_ff, configs.dropout)

        self.stage2_embed = nn.Conv1d(c, c * 2, kernel_size=5, stride=2, padding=2)
        self.stage2_block = CDATBlock(c * 2, configs.n_heads, configs.d_ff, configs.dropout)

        self.stage3_embed = nn.Conv1d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.stage3_block = CDATBlock(c * 4, configs.n_heads, configs.d_ff, configs.dropout)

        self.stage4_embed = nn.Conv1d(c * 4, c * 8, kernel_size=3, stride=2, padding=1)
        self.stage4_block = CDATBlock(c * 8, configs.n_heads, configs.d_ff, configs.dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-Specific Heads
        final_dim = c * 8

        if self.task_name == 'AMC':
            self.amc_classifier = self._build_classifier(final_dim, configs.n_classes_amc, configs.dropout)
            
        if self.task_name == 'WTC':
            self.wtc_classifier = self._build_classifier(final_dim, configs.n_classes_wtc, configs.dropout)
            
        if self.task_name == 'SS':
            self.ss_classifier = nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, configs.n_classes_ss)
            )
            
        if self.task_name == 'AD':
            self.ad_projection = nn.Linear(final_dim, configs.enc_in)

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
        Extracts features using the CDAT stages.
        x_enc: [B, 2, L]
        """
        x = self.stage1_embed(x_enc)
        x = self.stage1_block(x)

        x = self.stage2_embed(x)
        x = self.stage2_block(x)

        x = self.stage3_embed(x)
        x = self.stage3_block(x)

        x = self.stage4_embed(x)
        x = self.stage4_block(x)
        return x # [B, C_final, L_final]

    def amc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat = self.pool(feat).flatten(1)
        return self.amc_classifier(feat)

    def wtc(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat = self.pool(feat).flatten(1)
        return self.wtc_classifier(feat)

    def ss(self, x_enc):
        feat = self.feature_extraction(x_enc)
        feat = self.pool(feat).flatten(1)
        return self.ss_classifier(feat)

    def ad(self, x_enc):
        # 1. 记录原始序列长度 L (128)
        L_ori = x_enc.shape[2] 
        
        # 2. 特征提取 -> 得到 [B, C_final, L_downsampled] (例如 [64, 512, 8])
        feat = self.feature_extraction(x_enc) 
        
        # 3. 映射到通道空间 (Linear 作用在最后一个维度)
        feat = feat.transpose(1, 2)           # [B, 8, 512]
        out = self.ad_projection(feat)        # [B, 8, 2]
        out = out.transpose(1, 2)            # [B, 2, 8]
        
        out = F.interpolate(out, size=L_ori, mode='linear', align_corners=False)
        return out # 返回 [B, 2, 128]，与输入 x_enc 形状一致

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