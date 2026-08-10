"""NMformer model for automatic modulation classification.

Reference:
    NMformer: A Transformer for Noisy Modulation Classification in Wireless
    Communication <https://arxiv.org/abs/2411.02428>

The original work feeds 224x224 RGB constellation diagrams into a ViT-B/16
backbone. This implementation converts IQ frames ``[B, 2, L]`` into
constellation images on-the-fly, then classifies them with the same ViT
hyperparameters used in the paper / ``repo/NMformer``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer


class NMformerConfig(PretrainedConfig):
    """Configuration for :class:`NMformerModel`.

    Defaults follow the paper and ``repo/NMformer`` (ViT-B/16 on 224x224
    constellation diagrams).
    """

    model_type: str = "nmformer"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        d_ff: int = 3072,
        n_heads: int = 12,
        n_layers: int = 12,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        constellation_scale: float = 2.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.img_size: int = img_size
        self.patch_size: int = patch_size
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.attention_dropout: float = attention_dropout
        self.constellation_scale: float = constellation_scale
        self.pretrained: bool = pretrained
        self.freeze_backbone: bool = freeze_backbone


def iq_to_constellation(
    x: torch.Tensor,
    img_size: int = 224,
    scale: float = 2.5,
) -> torch.Tensor:
    """Rasterize IQ samples into ImageNet-normalized RGB constellation images.

    Args:
        x: IQ tensor of shape ``[B, 2, L]``.
        img_size: Output spatial size (H = W = ``img_size``).
        scale: Constellation axis half-range used in the paper (default 2.5).

    Returns:
        Tensor of shape ``[B, 3, img_size, img_size]``.
    """
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected IQ input [B, 2, L], got {tuple(x.shape)}")

    B: int
    L: int
    B, _, L = x.shape
    device: torch.device = x.device
    dtype: torch.dtype = x.dtype

    I: torch.Tensor = x[:, 0, :]
    Q: torch.Tensor = x[:, 1, :]

    # Map I/Q in [-scale, scale] to pixel coordinates.
    px: torch.Tensor = (
        ((I / scale + 1.0) * 0.5 * (img_size - 1)).round().long().clamp(0, img_size - 1)
    )
    # Flip Q so that positive Q appears toward the top of the image.
    py: torch.Tensor = (
        ((1.0 - (Q / scale + 1.0) * 0.5) * (img_size - 1))
        .round()
        .long()
        .clamp(0, img_size - 1)
    )

    batch_idx: torch.Tensor = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)
    )
    flat_idx: torch.Tensor = (
        batch_idx * (img_size * img_size) + py.reshape(-1) * img_size + px.reshape(-1)
    )
    ones: torch.Tensor = torch.ones(B * L, device=device, dtype=dtype)
    density: torch.Tensor = torch.zeros(
        B * img_size * img_size, device=device, dtype=dtype
    )
    density = density.index_add(0, flat_idx, ones)
    density = density.view(B, 1, img_size, img_size)

    # Normalize to [0, 1] and replicate to RGB (constellation diagram style).
    density = density / (density.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    img: torch.Tensor = density.repeat(1, 3, 1, 1)

    mean: torch.Tensor = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=dtype
    ).view(1, 3, 1, 1)
    std: torch.Tensor = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=dtype
    ).view(1, 3, 1, 1)
    return (img - mean) / std


def _is_vit_b16_compatible(config: NMformerConfig) -> bool:
    """Whether architecture matches torchvision ``vit_b_16`` defaults."""
    return (
        config.img_size == 224
        and config.patch_size == 16
        and config.d_model == 768
        and config.d_ff == 3072
        and config.n_heads == 12
        and config.n_layers == 12
    )


class NMformerModel(nn.Module):
    """`NMformer <https://arxiv.org/abs/2411.02428>`_ backbone.

    Converts IQ frames into constellation diagrams and classifies them with a
    Vision Transformer (ViT-B/16 hyperparameters by default).
    """

    config_class = NMformerConfig

    def __init__(self, config: NMformerConfig) -> None:
        super().__init__()
        self.config: NMformerConfig = config
        self.seq_len: int = config.seq_len
        self.n_classes: int = config.n_classes
        self.img_size: int = config.img_size
        self.constellation_scale: float = config.constellation_scale
        self.d_model: int = config.d_model

        self.backbone: VisionTransformer = VisionTransformer(
            image_size=config.img_size,
            patch_size=config.patch_size,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            hidden_dim=config.d_model,
            mlp_dim=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            num_classes=config.n_classes,
        )

        if config.pretrained and _is_vit_b16_compatible(config):
            self._load_vit_b16_pretrained()

        if config.freeze_backbone:
            self._freeze_backbone()

    def _load_vit_b16_pretrained(self) -> None:
        """Load ImageNet-pretrained ViT-B/16 weights (as in ``repo/NMformer``)."""
        ref: VisionTransformer = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        state: dict = ref.state_dict()
        # Drop classification head; keep backbone / encoder weights.
        drop_keys = [k for k in state if k.startswith("heads.")]
        for k in drop_keys:
            del state[k]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        # ``heads`` is intentionally re-initialized for ``n_classes``.
        _ = missing, unexpected

    def _freeze_backbone(self) -> None:
        """Freeze all parameters except the classification head (fine-tune mode)."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("heads.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: IQ tensor of shape ``[B, 2, seq_len]``.

        Returns:
            Logits of shape ``[B, n_classes]``.
        """
        images: torch.Tensor = iq_to_constellation(
            x, img_size=self.img_size, scale=self.constellation_scale
        )
        return self.backbone(images)
