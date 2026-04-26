import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助模块 (RevIN) ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x: torch.FloatTensor, mode: str) -> torch.FloatTensor:
        if mode == 'norm':
            self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.stdev + self.mean
        return x

# --- ModernTCN 核心组件 ---

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, groups: int, small_kernel: int) -> None:
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        self.lkb_origin = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        if small_kernel is not None:
            self.small_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out


class Modern_Block(nn.Module):
    def __init__(self, large_size: int, small_size: int, dmodel: int, dff: int, nvars: int, drop: float = 0.1) -> None:
        super(Modern_Block, self).__init__()
        
        # Depthwise Large Kernel Conv (空间建模)
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel, out_channels=nvars * dmodel,
            kernel_size=large_size, stride=1, groups=nvars * dmodel,
            small_kernel=small_size
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN1 (变量内建模 - Variable-wise)
        self.ffn1 = nn.Sequential(
            nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=nvars),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=nvars),
            nn.Dropout(drop)
        )

        # ConvFFN2 (维度内建模 - Dimension-wise)
        self.ffn2 = nn.Sequential(
            nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=dmodel),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=dmodel),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: [B, M, D, N] (Batch, Variables, Dimensions, Length)
        res = x
        B, M, D, N = x.shape

        # 1. Depthwise Conv
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        
        # 2. BatchNorm
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M * D, N)

        # 3. FFN1
        x = self.ffn1(x)
        
        # 4. FFN2 (Dimension/Channel Mixing)
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2(x)
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)

        x = x + res
        return x


class Modern_Stage(nn.Module):
    def __init__(self, num_blocks: int, ffn_ratio: int, large_size: int, small_size: int, 
                 dmodel: int, nvars: int, drop: float) -> None:
        super(Modern_Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList([
            Modern_Block(large_size, small_size, dmodel, d_ffn, nvars, drop) 
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for block in self.blocks:
            x = block(x)
        return x

# --- ModernTCN ---

class model(nn.Module):
    def __init__(self, configs) -> None:
        super(model, self).__init__()
        
        # 基础配置
        self.seq_len = configs.seq_len
        self.n_vars = configs.input_channels
        self.n_classes = configs.n_classes

        # 记录参数用于 forward 中的 Padding
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.downsample_ratio = configs.downsample_ratio
        
        # RevIN 归一化层 (可选)
        self.revin = RevIN(self.n_vars) if getattr(configs, 'revin', False) else None

        # --- Stem & Downsampling ---
        self.downsample_layers = nn.ModuleList()
        
        # Stem Layer
        stem = nn.Sequential(
            nn.Conv1d(1, configs.dims[0], kernel_size=configs.patch_size, stride=configs.patch_stride),
            nn.BatchNorm1d(configs.dims[0])
        )
        self.downsample_layers.append(stem)

        # Downsampling Layers
        for i in range(len(configs.num_blocks) - 1):
            down = nn.Sequential(
                nn.BatchNorm1d(configs.dims[i]),
                nn.Conv1d(configs.dims[i], configs.dims[i+1], 
                          kernel_size=configs.downsample_ratio, stride=configs.downsample_ratio)
            )
            self.downsample_layers.append(down)

        self.stages = nn.ModuleList()
        for i in range(len(configs.num_blocks)):
            stage = Modern_Stage(
                num_blocks=configs.num_blocks[i],
                ffn_ratio=configs.ffn_ratio,
                large_size=configs.large_size[i],
                small_size=configs.small_size[i],
                dmodel=configs.dims[i],
                nvars=self.n_vars,
                drop=configs.dropout
            )
            self.stages.append(stage)

        # --- Classification Head ---
        patch_num = self.seq_len // self.patch_stride
        final_len = patch_num // (self.downsample_ratio ** (len(configs.num_blocks) - 1))
        self.head_nf = configs.dims[-1] * final_len
        
        self.class_dropout = nn.Dropout(configs.class_dropout)
        self.classifier = nn.Linear(self.n_vars * self.head_nf, self.n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Input x: [Batch, Vars, Seq_len] 
        """
        
        # 1. 实例归一化 (RevIN)
        if self.revin:
            x = self.revin(x, 'norm')

        # 2. Backbone
        # [B, M, 1, L] 
        x = x.unsqueeze(-2)
        
        for i in range(len(self.stages)):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            
            # Stem 或 下采样处理
            x = self.downsample_layers[i](x)
            
            # 还原形状并进入 Modern Stage
            _, D_new, N_new = x.shape
            x = x.reshape(B, M, D_new, N_new)
            x = self.stages[i](x)

        # 3. ModernTCN Classification
        # 激活层
        x = F.gelu(x)
        # Dropout
        x = self.class_dropout(x)
        # [B, M, D, N] -> [B, M*D*N]
        x = x.reshape(x.shape[0], -1)
        y = self.classifier(x)

        return y

    def structural_reparam(self) -> None:
        """
        结构化重参数化：将训练时的多分支卷积融合为推理时的单路卷积
        """
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                # 如果定义了融合逻辑，此处调用
                pass