"""SpectrumFM model for automatic modulation classification.

Reference:
    SpectrumFM: A Foundation Model for Intelligent Spectrum Management
    <https://ieeexplore.ieee.org/document/11301740>

This ports the AMC fine-tuning backbone ``ConformerClassifier`` from
``repo/SpectrumFM/Model/model.py`` (used in ``amc.py``).

Paper / code defaults:
    input_dim=2, d_model=256, n_heads=4, n_layers=16, d_ff=512, seq_len=128.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig


# ---------------------------------------------------------------------------
# Custom LoRA (as in ``repo/SpectrumFM`` ConformerClassifier)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.r: int = r
        self.alpha: float = alpha
        self.scaling: float = alpha / r
        self.linear: nn.Linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A: nn.Parameter = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B: nn.Parameter = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 4,
        alpha: float = 1.0,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.r: int = r
        self.alpha: float = alpha
        self.scaling: float = alpha / r
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.lora_A: nn.Conv1d = nn.Conv1d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B: nn.Conv1d = nn.Conv1d(r, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _get_parent_module(model: nn.Module, full_name: str) -> nn.Module:
    parts: List[str] = full_name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    return model


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int = 4,
    alpha: float = 1.0,
) -> None:
    """Replace Linear / Conv1d modules whose names contain any target substring."""
    for name, _module in list(model.named_modules()):
        if not any(target in name for target in target_modules):
            continue
        parent = _get_parent_module(model, name)
        attr_name: str = name.split(".")[-1]
        orig_module = getattr(parent, attr_name)

        if isinstance(orig_module, nn.Linear):
            lora_module = LoRALinear(
                orig_module.in_features,
                orig_module.out_features,
                r=r,
                alpha=alpha,
                bias=orig_module.bias is not None,
            )
            lora_module.linear.weight.data = orig_module.weight.data.clone()
            if orig_module.bias is not None:
                lora_module.linear.bias.data = orig_module.bias.data.clone()
            setattr(parent, attr_name, lora_module)

        elif isinstance(orig_module, nn.Conv1d) and orig_module.groups == 1:
            lora_module = LoRAConv1d(
                orig_module.in_channels,
                orig_module.out_channels,
                orig_module.kernel_size[0],
                r=r,
                alpha=alpha,
                stride=orig_module.stride[0],
                padding=orig_module.padding[0],
                dilation=orig_module.dilation[0],
                bias=orig_module.bias is not None,
            )
            lora_module.conv.weight.data = orig_module.weight.data.clone()
            if orig_module.bias is not None:
                lora_module.conv.bias.data = orig_module.bias.data.clone()
            setattr(parent, attr_name, lora_module)


# ---------------------------------------------------------------------------
# Conformer building blocks
# ---------------------------------------------------------------------------


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class RelativePositionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_len: int = 512) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.max_len: int = max_len
        self.query: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relative_positions: nn.Parameter = nn.Parameter(
            torch.randn(2 * max_len - 1, num_heads)
        )
        self.output: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(0.2)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        head_dim: int = self.embed_dim // self.num_heads

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, head_dim)

        attention_scores: torch.Tensor = torch.einsum("bqhd,bkhd->bhqk", Q, K)
        attention_scores = attention_scores / (head_dim**0.5)

        position_indices = torch.arange(seq_len, device=x.device).unsqueeze(
            0
        ) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        position_indices = (position_indices + self.max_len - 1).clamp(
            min=0, max=2 * self.max_len - 2
        )
        relative_position_embedding = self.relative_positions[position_indices]
        relative_position_embedding = relative_position_embedding.permute(2, 0, 1)
        relative_position_embedding = relative_position_embedding.unsqueeze(0)
        attention_scores = attention_scores + relative_position_embedding

        if mask is not None:
            attn_mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                mask, "b j -> b () () j"
            )
            mask_value = -torch.finfo(attention_scores.dtype).max
            attention_scores = attention_scores.masked_fill(attn_mask == 0, mask_value)

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
        output = output.contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.dropout(self.output(output))


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(model_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, model_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.gelu: nn.GELU = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pointwise1: nn.Conv1d = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size=1
        )
        self.depthwise: nn.Conv1d = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            padding=kernel_size // 2,
        )
        self.pointwise2: nn.Conv1d = nn.Conv1d(
            out_channels, out_channels, kernel_size=1
        )
        self.glu: GLU = GLU(dim=1)
        self.swish: Swish = Swish()
        self.bn: nn.BatchNorm1d = nn.BatchNorm1d(out_channels)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.glu(self.pointwise1(x))
        x = self.swish(self.bn(self.depthwise(x)))
        x = self.dropout(self.pointwise2(x))
        return x.permute(0, 2, 1)


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, ff_hidden_dim: int, max_len: int
    ) -> None:
        super().__init__()
        self.attention: RelativePositionAttention = RelativePositionAttention(
            model_dim, num_heads, max_len
        )
        self.feed_forward1: FeedForward = FeedForward(model_dim, ff_hidden_dim)
        self.feed_forward2: FeedForward = FeedForward(model_dim, ff_hidden_dim)
        self.conv: DepthwiseConv = DepthwiseConv(model_dim, model_dim)
        self.norm1: nn.LayerNorm = nn.LayerNorm(model_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(model_dim)
        self.norm3: nn.LayerNorm = nn.LayerNorm(model_dim)
        self.norm4: nn.LayerNorm = nn.LayerNorm(model_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm1(x + 0.5 * self.feed_forward1(x))
        x = self.norm2(x + self.attention(x, mask))
        x = self.norm3(x + self.conv(x))
        x = self.norm4(x + 0.5 * self.feed_forward2(x))
        return x


class InputProjection(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.input_proj: nn.Conv1d = nn.Conv1d(input_dim, model_dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_hidden_dim: int,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.model_dim: int = model_dim
        self.layers: nn.ModuleList = nn.ModuleList(
            [InputProjection(input_dim, model_dim)]
            + [
                ConformerEncoderLayer(model_dim, num_heads, ff_hidden_dim, max_len)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ---------------------------------------------------------------------------
# Config / Model
# ---------------------------------------------------------------------------


class SpectrumFMConfig(PretrainedConfig):
    """Configuration for :class:`SpectrumFMModel`.

    Defaults follow the SpectrumFM paper / ``amc.py`` ConformerClassifier call.
    """

    model_type: str = "spectrumfm"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        enc_in: int = 2,
        d_model: int = 256,
        d_ff: int = 512,
        n_heads: int = 4,
        n_layers: int = 16,
        dropout: float = 0.2,
        max_len: int = 1024,
        is_LORA: bool = False,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.enc_in: int = enc_in
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.max_len: int = max_len
        self.is_LORA: bool = is_LORA
        self.lora_r: int = lora_r
        self.lora_alpha: float = lora_alpha
        self.pretrained_path: Optional[str] = pretrained_path
        self.freeze_encoder: bool = freeze_encoder


class SpectrumFMModel(nn.Module):
    """SpectrumFM Conformer classifier for AMC.

    Input shape: ``[B, 2, seq_len]`` (I/Q). Internally converted to ``[B, L, 2]``.
    Output shape: ``[B, n_classes]``.
    """

    config_class = SpectrumFMConfig

    def __init__(self, config: SpectrumFMConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})."
            )

        self.config: SpectrumFMConfig = config
        self.seq_len: int = config.seq_len
        self.n_classes: int = config.n_classes
        self.d_model: int = config.d_model

        self.encoder: ConformerEncoder = ConformerEncoder(
            input_dim=config.enc_in,
            model_dim=config.d_model,
            num_heads=config.n_heads,
            num_layers=config.n_layers,
            ff_hidden_dim=config.d_ff,
            max_len=config.max_len,
        )

        if config.pretrained_path:
            state = torch.load(config.pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)

        if config.is_LORA:
            inject_lora(
                self.encoder,
                target_modules=["query", "key", "value", "output", "fc1", "fc2"],
                r=config.lora_r,
                alpha=config.lora_alpha,
            )
            if config.freeze_encoder:
                for name, param in self.encoder.named_parameters():
                    if "lora_" not in name:
                        param.requires_grad = False

        self.gru: nn.GRU = nn.GRU(
            config.d_model,
            config.d_model,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.classifier: nn.Linear = nn.Linear(config.d_model * 2, config.n_classes)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: IQ tensor ``[B, 2, L]``.
            mask: Optional attention mask for relative-position attention.
        """
        # Project library layout [B, 2, L] -> SpectrumFM layout [B, L, 2]
        if x.dim() == 3 and x.size(1) == self.config.enc_in:
            x = x.permute(0, 2, 1).contiguous()

        x = self.encoder(x, mask)
        out, _hn = self.gru(x)
        x = self.dropout(out[:, -1, :])
        return self.classifier(x)
