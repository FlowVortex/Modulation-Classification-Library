"""IOformer (IQFormer) model for automatic modulation classification.

Reference:
    IQFormer from FoundWSR
    ``repo/FoundWSR/foundwsr/models/IQFormer``

Reimplemented without ``timm`` (local ``DropPath`` / ``trunc_normal_``).
Defaults follow ``config.yaml`` in the original IQFormer folder.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


# ---------------------------------------------------------------------------
# timm replacements
# ---------------------------------------------------------------------------


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Truncated normal initialization (replaces ``timm.models.layers.trunc_normal_``)."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Stochastic depth (replaces ``timm.models.layers.DropPath``)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob: float = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob: float = 1.0 - self.drop_prob
        shape: Tuple[int, ...] = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor: torch.Tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor = random_tensor.div(keep_prob)
        return x * random_tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def stemIQ(in_chs: int, out_chs: int) -> nn.Sequential:
    """IQ stem: depthwise-grouped Conv1d + BN. Output ``[B, out_chs // 2, L]``."""
    return nn.Sequential(
        nn.Conv1d(
            in_chs,
            out_chs // 2,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_chs,
        ),
        nn.BatchNorm1d(out_chs // 2),
    )


def stemSTFT(f: int, in_chs: int, out_chs: int) -> nn.Sequential:
    """STFT stem: Conv2d over frequency. Output ``[B, out_chs // 2, 1, T]``."""
    return nn.Sequential(
        nn.Conv2d(
            in_chs,
            out_chs // 2,
            kernel_size=(f, 1),
            stride=1,
            groups=in_chs,
        ),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
    )


class Embedding(nn.Module):
    """Downsampling patch embedding between stages. ``[B, C, D] -> [B, C', D']``."""

    def __init__(
        self,
        patch_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.proj: nn.Conv1d = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm: nn.Module = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ConvEncoder_IQ(nn.Module):
    """Depthwise / pointwise Conv encoder block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dwconv: nn.Conv1d = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(dim)
        self.pwconv1: nn.Conv1d = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act: nn.GELU = nn.GELU()
        self.pwconv2: nn.Conv1d = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.use_layer_scale: bool = use_layer_scale
        if use_layer_scale:
            self.layer_scale: nn.Parameter = nn.Parameter(
                torch.ones(dim).unsqueeze(-1), requires_grad=True
            )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        if self.use_layer_scale:
            return residual + self.drop_path(self.layer_scale * x)
        return residual + self.drop_path(x)


class FCN(nn.Module):
    """Pointwise FCN (1x1 conv MLP)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(in_features)
        self.fc1: nn.Conv1d = nn.Conv1d(in_features, hidden_features, 1)
        self.act: nn.Module = act_layer()
        self.fc2: nn.Conv1d = nn.Conv1d(hidden_features, out_features, 1)
        self.drop: nn.Dropout = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(self.norm1(x)))
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """Efficient Additive Attention (kept original class name spelling)."""

    def __init__(
        self, in_dims: int = 512, token_dim: int = 256, num_heads: int = 2
    ) -> None:
        super().__init__()
        self.to_query: nn.Linear = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key: nn.Linear = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g: nn.Parameter = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor: float = token_dim**-0.5
        self.Proj: nn.Linear = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final: nn.Linear = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor = F.normalize(self.to_query(x), dim=-1)
        key: torch.Tensor = F.normalize(self.to_key(x), dim=-1)
        query_weight: torch.Tensor = query @ self.w_g
        A: torch.Tensor = F.normalize(query_weight * self.scale_factor, dim=1)
        G: torch.Tensor = torch.sum(A * query, dim=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])
        out: torch.Tensor = self.Proj(G * key) + query
        return self.final(out)


class LocalRepresentation(nn.Module):
    """Local 3x3 depthwise + pointwise representation."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dwconv: nn.Conv1d = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(dim)
        self.pwconv1: nn.Conv1d = nn.Conv1d(dim, dim, kernel_size=1)
        self.act: nn.GELU = nn.GELU()
        self.pwconv2: nn.Conv1d = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.use_layer_scale: bool = use_layer_scale
        if use_layer_scale:
            self.layer_scale: nn.Parameter = nn.Parameter(
                torch.ones(dim).unsqueeze(-1), requires_grad=True
            )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        if self.use_layer_scale:
            return residual + self.drop_path(self.layer_scale * x)
        return residual + self.drop_path(x)


class Fusion(nn.Module):
    """Fuse IQ stem features with STFT stem features."""

    def __init__(self, input_chanel: int, drop: float) -> None:
        super().__init__()
        self.Conv: nn.Sequential = nn.Sequential(
            nn.Conv1d(input_chanel, input_chanel * 2, 1),
            nn.BatchNorm1d(input_chanel * 2),
            nn.GELU(),
            nn.Conv1d(input_chanel * 2, input_chanel * 2, 1),
        )
        self.drop: nn.Dropout = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, stft_feat: torch.Tensor) -> torch.Tensor:
        return self.drop(self.Conv(torch.cat((x, stft_feat), dim=1)))


class IQFormer_Encoder(nn.Module):
    """Local representation + Efficient Additive Attention + FCN."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.local_representation: LocalRepresentation = LocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0.0, use_layer_scale=True
        )
        self.attn: EfficientAdditiveAttnetion = EfficientAdditiveAttnetion(
            in_dims=dim, token_dim=dim, num_heads=1
        )
        self.linear: FCN = FCN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.use_layer_scale: bool = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1: nn.Parameter = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1),
                requires_grad=True,
            )
            self.layer_scale_2: nn.Parameter = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1),
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_representation(x)
        attn_out: torch.Tensor = self.attn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * attn_out)
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.linear(x))
        return x


def build_stage(
    dim: int,
    index: int,
    layers: Sequence[int],
    mlp_ratio: float = 4.0,
    act_layer: type = nn.GELU,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    vit_num: int = 1,
) -> nn.Sequential:
    """Build one IQFormer stage (ConvEncoder + trailing IQFormer_Encoder blocks)."""
    blocks: List[nn.Module] = []
    total_blocks: int = max(sum(layers) - 1, 1)
    for block_idx in range(layers[index]):
        block_dpr: float = (
            drop_path_rate * (block_idx + sum(layers[:index])) / total_blocks
        )
        if layers[index] - block_idx <= vit_num:
            blocks.append(
                IQFormer_Encoder(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            blocks.append(
                ConvEncoder_IQ(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3)
            )
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# Config / Model
# ---------------------------------------------------------------------------


class IOformerConfig(PretrainedConfig):
    """Configuration for :class:`IOformerModel`.

    Defaults follow ``repo/FoundWSR/.../IQFormer/config.yaml``.
    """

    model_type: str = "ioformer"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        layers: Optional[List[int]] = None,
        embed_dims: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        vit_num: int = 1,
        down_patch_size: int = 5,
        down_stride: int = 32,
        down_pad: int = 1,
        stft_nperseg: int = 31,
        stft_noverlap: int = 30,
        stft_nfft: int = 128,
        stft_freq_bins: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.layers: List[int] = list(layers) if layers is not None else [2, 3, 2]
        self.embed_dims: List[int] = (
            list(embed_dims) if embed_dims is not None else [64, 64, 64]
        )
        self.mlp_ratio: float = mlp_ratio
        self.dropout: float = dropout
        self.drop_path_rate: float = drop_path_rate
        self.use_layer_scale: bool = use_layer_scale
        self.layer_scale_init_value: float = layer_scale_init_value
        self.vit_num: int = vit_num
        self.down_patch_size: int = down_patch_size
        self.down_stride: int = down_stride
        self.down_pad: int = down_pad
        self.stft_nperseg: int = stft_nperseg
        self.stft_noverlap: int = stft_noverlap
        self.stft_nfft: int = stft_nfft
        self.stft_freq_bins: int = stft_freq_bins


class IOformerModel(nn.Module):
    """IQFormer backbone for IQ + STFT dual-branch modulation classification.

    Input: ``[B, 2, seq_len]`` (I/Q). STFT is computed internally from the I channel.
    Output: ``[B, n_classes]``.
    """

    config_class = IOformerConfig

    def __init__(self, config: IOformerConfig) -> None:
        super().__init__()
        self.config: IOformerConfig = config
        self.seq_len: int = config.seq_len
        self.n_classes: int = config.n_classes
        self.layers: List[int] = list(config.layers)
        self.embed_dims: List[int] = list(config.embed_dims)
        self.stft_nperseg: int = config.stft_nperseg
        self.stft_noverlap: int = config.stft_noverlap
        self.stft_nfft: int = config.stft_nfft
        self.stft_freq_bins: int = config.stft_freq_bins

        self.BN: nn.BatchNorm1d = nn.BatchNorm1d(2)
        self.BN_stft: nn.BatchNorm2d = nn.BatchNorm2d(1)
        self.patch_embedIQ: nn.Sequential = stemIQ(2, self.embed_dims[0] // 4)
        self.patch_embedSTFT: nn.Sequential = stemSTFT(
            self.stft_freq_bins, 1, self.embed_dims[0] // 4
        )
        self.fusion: Fusion = Fusion(self.embed_dims[0] // 4, config.dropout)

        network: List[nn.Module] = []
        for i in range(len(self.layers)):
            network.append(
                build_stage(
                    self.embed_dims[i],
                    i,
                    self.layers,
                    mlp_ratio=config.mlp_ratio,
                    act_layer=nn.GELU,
                    drop_rate=config.dropout,
                    drop_path_rate=config.drop_path_rate,
                    use_layer_scale=config.use_layer_scale,
                    layer_scale_init_value=config.layer_scale_init_value,
                    vit_num=config.vit_num,
                )
            )
            if i >= len(self.layers) - 1:
                break
            if self.embed_dims[i] != self.embed_dims[i + 1]:
                network.append(
                    Embedding(
                        patch_size=config.down_patch_size,
                        stride=config.down_stride,
                        padding=config.down_pad,
                        in_chans=self.embed_dims[i],
                        embed_dim=self.embed_dims[i + 1],
                    )
                )
        self.network: nn.ModuleList = nn.ModuleList(network)

        self.patch_LSTM: nn.LSTM = nn.LSTM(
            input_size=self.embed_dims[0] // 2,
            hidden_size=self.embed_dims[0] // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=config.dropout if config.dropout > 0 else 0.0,
        )

        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(self.embed_dims[-1])
        self.head: nn.Module = (
            nn.Linear(self.embed_dims[-1], self.n_classes)
            if self.n_classes > 0
            else nn.Identity()
        )
        self.globalavgpool: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude STFT of the I-channel.

        Args:
            x: IQ tensor ``[B, 2, L]``.

        Returns:
            Spectrogram ``[B, 1, freq_bins, T]``.
        """
        i_ch: torch.Tensor = x[:, 0, :]
        hop_length: int = max(1, self.stft_nperseg - self.stft_noverlap)
        window: torch.Tensor = torch.blackman_window(
            self.stft_nperseg, device=x.device, dtype=x.dtype
        )
        spec: torch.Tensor = torch.stft(
            i_ch,
            n_fft=self.stft_nfft,
            hop_length=hop_length,
            win_length=self.stft_nperseg,
            window=window,
            center=True,
            return_complex=True,
        )
        mag: torch.Tensor = spec.abs()[:, : self.stft_freq_bins, :]
        return mag.unsqueeze(1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.network:
            x = block(x)
        return x

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            x: IQ ``[B, 2, L]``, or ``(iq, stft)`` where stft is ``[B, 1, F, T]``.
        """
        if isinstance(x, tuple):
            stft_x: torch.Tensor = x[1]
            x = x[0]
        else:
            stft_x = self.compute_stft(x)

        x = self.BN(x)
        stft_x = self.BN_stft(stft_x)
        x = self.patch_embedIQ(x)
        stft_x = self.patch_embedSTFT(stft_x).squeeze(2)
        # Align STFT time axis with IQ stem length (torch.stft frame count may differ).
        if stft_x.shape[-1] != x.shape[-1]:
            stft_x = F.interpolate(
                stft_x, size=x.shape[-1], mode="linear", align_corners=False
            )
        x = self.fusion(x, stft_x)
        x, _ = self.patch_LSTM(x.permute(0, 2, 1))
        x = self.forward_tokens(x.permute(0, 2, 1))
        x = self.norm(x)
        return self.head(self.globalavgpool(x))
