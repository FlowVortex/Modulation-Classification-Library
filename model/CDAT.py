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

        Q = self.q_proj(x)  # [B, C, L]
        K = self.k_proj(x)  # [B, C, L]
        V = self.v_proj(x)  # [B, C, L]

        head_outputs = []
        for i in range(self.n_heads):
            # 分割多头
            start, end = i * self.d_head, (i + 1) * self.d_head
            qi, ki, vi = Q[:, start:end, :], K[:, start:end, :], V[:, start:end, :]

            # Apos
            # [B, d_head, L] -> [B, L, d_head]
            qi_t = qi.transpose(1, 2)
            # attn = Softmax(Qi^T * Ki / sqrt(dk))
            attn_weight = torch.matmul(qi_t, ki) * (self.d_head**-0.5)
            attn_weight = F.softmax(attn_weight, dim=-1)  # [B, L, L]
            # apos = attn * Vi^T -> [B, L, d_head]
            apos = torch.matmul(attn_weight, vi.transpose(1, 2))
            apos = apos.transpose(1, 2)  # [B, d_head, L]

            # Ac = Sigmoid(DSConv(Qi)) * Vi (Hadamard product)
            ac_weight = self.sigmoid(self.ac_convs[i](qi))  # [B, d_head, L]
            ac = ac_weight * vi  # [B, d_head, L]

            # CDAi = Apos + Ac
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
        # x shape: [B, C, L]
        B, C, L = x.shape

        # attn
        residual = x
        x = x.transpose(1, 2)  # [B, L, C]
        x = self.ln1(x)
        x = x.transpose(1, 2)  # [B, C, L]
        x = residual + self.dropout(self.attn(x))

        # FFN
        residual = x
        x = x.transpose(1, 2)  # [B, L, C]
        x = self.ln2(x)
        x = self.ffn(x)
        x = x.transpose(1, 2)  # [B, C, L]
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

        # channel
        c = configs.d_model

        # Stage 1: Kernel 7, Stride 2
        self.stage1_embed = nn.Conv1d(2, c, kernel_size=7, stride=2, padding=3)
        self.stage1_block = CDATBlock(c, configs.n_heads, configs.d_ff)

        # Stage 2: Kernel 5, Stride 2
        self.stage2_embed = nn.Conv1d(c, c * 2, kernel_size=5, stride=2, padding=2)
        self.stage2_block = CDATBlock(c * 2, configs.n_heads, configs.d_ff)

        # Stage 3: Kernel 3, Stride 2
        self.stage3_embed = nn.Conv1d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.stage3_block = CDATBlock(c * 4, configs.n_heads, configs.d_ff)

        # Stage 4: Kernel 3, Stride 2
        self.stage4_embed = nn.Conv1d(c * 4, c * 8, kernel_size=3, stride=2, padding=1)
        self.stage4_block = CDATBlock(c * 8, configs.n_heads, configs.d_ff)

        # classifier: Pooling -> Linear
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(c * 8, configs.n_classes)

    def forward(self, x):
        # x: [B, 2, L]

        x = self.stage1_embed(x)
        x = self.stage1_block(x)

        x = self.stage2_embed(x)
        x = self.stage2_block(x)

        x = self.stage3_embed(x)
        x = self.stage3_block(x)

        x = self.stage4_embed(x)
        x = self.stage4_block(x)

        x = self.pool(x).flatten(1)  # [B, C_final]
        return self.classifier(x)
