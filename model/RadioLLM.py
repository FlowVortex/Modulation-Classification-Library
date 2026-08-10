"""RadioLLM model for automatic modulation classification.

Reference:
    RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid
    Prompt and Token Reprogrammings <https://arxiv.org/abs/2501.17888>

This file ports the **classification** path from ``repo/RadioLLM``, including
dataset description prompts and the hybrid soft/hard prompt template used in
``RadioLLM.classific``.
"""

from __future__ import annotations

import math
from math import sqrt
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    PretrainedConfig,
)

try:
    from peft import LoraConfig, get_peft_model

    _PEFT_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Supporting modules (from ``repo/RadioLLM``)
# ---------------------------------------------------------------------------


class ComplexConv(nn.Module):
    """Complex-valued 1D convolution used by the frequency branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_re: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv_im: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real: torch.Tensor = x[:, 0 : x.shape[1] // 2, :]
        x_img: torch.Tensor = x[:, x.shape[1] // 2 :, :]
        real: torch.Tensor = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary: torch.Tensor = self.conv_re(x_img) + self.conv_im(x_real)
        return torch.cat((real, imaginary), dim=1)


class ReplicationPad1d(nn.Module):
    def __init__(self, padding: Tuple[int, int]) -> None:
        super().__init__()
        self.padding: Tuple[int, int] = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding: Tensor = (
            input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        )
        return torch.cat([input, replicate_padding], dim=-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe: torch.Tensor = torch.zeros(max_len, d_model).float()
        position: torch.Tensor = torch.arange(0, max_len).float().unsqueeze(1)
        div_term: torch.Tensor = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pe: nn.Parameter = nn.Parameter(
            torch.zeros(1, max_len, d_model), requires_grad=True
        )
        pe: torch.Tensor = torch.zeros(max_len, d_model).float()
        position: torch.Tensor = torch.arange(0, max_len).float().unsqueeze(1)
        div_term: torch.Tensor = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.data.copy_(pe.unsqueeze(0).float())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.pe[:, offset : offset + x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding: int = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv: nn.Conv1d = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(
        self, d_model: int, patch_len: int, stride: int, dropout: float
    ) -> None:
        super().__init__()
        self.patch_len: int = patch_len
        self.stride: int = stride
        self.padding_patch_layer: ReplicationPad1d = ReplicationPad1d((0, stride))
        self.value_embedding: TokenEmbedding = TokenEmbedding(patch_len, d_model)
        self.position_embedding: LearnablePositionalEmbedding = (
            LearnablePositionalEmbedding(d_model=d_model)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        n_vars: int = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class Normalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.affine: bool = affine
        self.subtract_last: bool = subtract_last
        self.non_norm: bool = non_norm
        if self.affine:
            self.affine_weight: nn.Parameter = nn.Parameter(torch.ones(num_features))
            self.affine_bias: nn.Parameter = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class high_freq_extract(nn.Module):
    def __init__(self, in_channels: int = 128, out_channels: int = 768) -> None:
        super().__init__()
        self.conv1: ComplexConv = ComplexConv(
            in_channels=in_channels,
            out_channels=out_channels // 8,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm1: nn.BatchNorm1d = nn.BatchNorm1d(num_features=out_channels // 4)
        self.conv2: ComplexConv = ComplexConv(
            in_channels=out_channels // 8,
            out_channels=out_channels // 4,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm2: nn.BatchNorm1d = nn.BatchNorm1d(num_features=out_channels // 2)
        self.conv3: ComplexConv = ComplexConv(
            in_channels=out_channels // 4,
            out_channels=out_channels // 2,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm3: nn.BatchNorm1d = nn.BatchNorm1d(num_features=out_channels)
        self.convres: ComplexConv = ComplexConv(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=3,
            padding=1,
        )
        self.batchnormres: nn.BatchNorm1d = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, sgn: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = F.relu(self.conv1(sgn))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        res: torch.Tensor = F.relu(self.convres(sgn))
        res = self.batchnormres(res)
        return x + res


class High_freq_conv_3layer(nn.Module):
    """Frequency Attuned Fusion (FAF) high-frequency extractor."""

    def __init__(
        self,
        d_model: int,
        out_channel: int,
        patch_len: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len: int = patch_len
        self.stride: int = stride
        self.padding_patch_layer: ReplicationPad1d = ReplicationPad1d((0, stride))
        self.value_embedding: nn.Conv2d = nn.Conv2d(
            1,
            d_model,
            kernel_size=(1, patch_len),
            stride=(1, stride),
            padding=(0, stride),
            bias=False,
        )
        self.high_freq_exter: high_freq_extract = high_freq_extract(
            in_channels=d_model, out_channels=d_model * 2
        )
        self.high_freq_exter2: high_freq_extract = high_freq_extract(
            in_channels=d_model, out_channels=out_channel * 2
        )
        self.high_freq_exter3: high_freq_extract = high_freq_extract(
            in_channels=out_channel, out_channels=out_channel * 2
        )
        self.pool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=2)
        self.drop: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding_patch_layer(x)
        x = torch.unsqueeze(x, 1)
        x = self.value_embedding(x)
        x = self.drop(x)
        x = einops.rearrange(x, "b d n l -> b (n d) l")
        x = self.pool(self.high_freq_exter(x))
        x = self.pool(self.high_freq_exter2(x))
        x = self.pool(self.high_freq_exter3(x))
        return x


class ReprogrammingLayer(nn.Module):
    """Token reprogramming that aligns signal patches with LLM word space."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_llm: Optional[int] = None,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection: nn.Linear = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection: nn.Linear = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection: nn.Linear = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection: nn.Linear = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads: int = n_heads
        self.dropout: nn.Dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,
        source_embedding: torch.Tensor,
        value_embedding: torch.Tensor,
    ) -> torch.Tensor:
        B: int
        L: int
        B, L, _ = target_embedding.shape
        S: int = source_embedding.shape[0]
        H: int = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out: torch.Tensor = self.reprogramming(
            target_embedding, source_embedding, value_embedding
        )
        return self.out_projection(out.reshape(B, L, -1))

    def reprogramming(
        self,
        target_embedding: torch.Tensor,
        source_embedding: torch.Tensor,
        value_embedding: torch.Tensor,
    ) -> torch.Tensor:
        B, L, H, E = target_embedding.shape
        scale: float = 1.0 / sqrt(E)
        scores: torch.Tensor = torch.einsum(
            "blhe,she->bhls", target_embedding, source_embedding
        )
        A: torch.Tensor = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value_embedding)


class AttentionFusion(nn.Module):
    """Fuse reprogrammed tokens with high-frequency features (FAF)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.query: nn.Linear = nn.Linear(in_channels, in_channels)
        self.key: nn.Linear = nn.Linear(in_channels, in_channels)
        self.value: nn.Linear = nn.Linear(in_channels, in_channels)

    def forward(self, enc_out: torch.Tensor, enc_out_sgn: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor = self.query(enc_out)
        key: torch.Tensor = self.key(enc_out_sgn)
        value: torch.Tensor = self.value(enc_out_sgn)
        attention_scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.in_channels**0.5)
        attention_weights: torch.Tensor = F.softmax(attention_scores, dim=-1)
        attended_features: torch.Tensor = torch.matmul(attention_weights, value)
        return enc_out + attended_features


def soft_hard_prompt(
    prompt_embeddings: torch.Tensor, prompt_word_embeddings: torch.Tensor, K: int
) -> torch.Tensor:
    """Hybrid soft/hard prompt: replace soft prompts by top-K word prototypes."""
    if K == 0:
        return prompt_embeddings
    B, LP, D = prompt_embeddings.shape
    sim: torch.Tensor = torch.einsum(
        "bld,vd->bv", prompt_embeddings, prompt_word_embeddings
    )
    _, topk_indices = torch.topk(sim, k=K, dim=-1)
    topk_indices_expanded: torch.Tensor = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(
        prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded
    )


# ---------------------------------------------------------------------------
# Text prompts synced from ``repo/RadioLLM/script/multi_task_pretrain.yaml``
# and the classification / forecast prompt template in ``radiollm.py``.
# ---------------------------------------------------------------------------

DATASET_DESCRIPTIONS: Dict[str, str] = {
    "RML2016": (
        "The RadioML 2016.10A dataset contains synthetic radar signals generated "
        "through software simulations. It consists of 11 different modulation types, "
        "including various forms of amplitude, frequency, and phase modulation. "
        "The signals are simulated under varying signal-to-noise ratios, mimicking "
        "real-world conditions. This dataset's synthetic nature allows for controlled "
        "experiments and reproducibility."
    ),
    "RML2016a": (
        "The RadioML 2016.10a is a comprehensive dataset for evaluating wireless "
        "signal modulation recognition algorithms. It is a vital resource in the "
        "field of cognitive radio and dynamic spectrum access, enabling efficient "
        "utilization of the electromagnetic spectrum."
    ),
    "RML2016b": (
        "The RadioML 2016.10A dataset contains synthetic radar signals generated "
        "through software simulations. It consists of 11 different modulation types, "
        "including various forms of amplitude, frequency, and phase modulation. "
        "The signals are simulated under varying signal-to-noise ratios, mimicking "
        "real-world conditions. This dataset's synthetic nature allows for controlled "
        "experiments and reproducibility."
    ),
    "RML2018": (
        "An extension of the 2016.10A version, the RadioML 2018.01A dataset "
        "introduces a larger variety of 24 modulation schemes and a wider range of "
        "signal-to-noise ratios. It incorporates more complex propagation effects, "
        "such as multipath and Doppler shifts, to better emulate realistic scenarios. "
        "This dataset's comprehensive signal diversity and challenging conditions "
        "make it suitable for advanced signal classification tasks."
    ),
    "RML2018a": (
        "An extension of the 2016.10A version, the RadioML 2018.01A dataset "
        "introduces a larger variety of 24 modulation schemes and a wider range of "
        "signal-to-noise ratios. It incorporates more complex propagation effects, "
        "such as multipath and Doppler shifts, to better emulate realistic scenarios. "
        "This dataset's comprehensive signal diversity and challenging conditions "
        "make it suitable for advanced signal classification tasks."
    ),
    "ADSB": (
        "Unlike the synthetic RadioML datasets, the ADS-B dataset consists of "
        "real-world radio frequency signals captured from aircraft transponders. "
        "It includes metadata such as aircraft position, altitude, and velocity, "
        "enabling applications in aircraft tracking and surveillance. The dataset's "
        "real-world nature introduces challenges like signal interference, noise, "
        "and dynamic conditions, making it valuable for evaluating practical signal "
        "processing and classification models."
    ),
    "WIFI": (
        "The WIFI dataset includes I/Q samples collected from a 16-node USRP X310 "
        "software defined radio (SDR) testbed as well as samples from 140 commercial "
        "off-the-shelf WiFi devices. The USRP testbed consists of 16 transmitting "
        "nodes that are identical USRP X310 radios sending IEEE 802.11a WiFi "
        "compliant data frames. The dataset is used to evaluate ORACLE's "
        "classification performance under static and dynamic channel conditions."
    ),
}

# Default finetune.py ``--content`` string (RadioML 2016.10a).
DEFAULT_DATASET_DESCRIPTION: str = DATASET_DESCRIPTIONS["RML2016a"]

# Task prompt lines from ``RadioLLM.classific`` / ``forecast``.
TASK_PROMPT_DENOISE: str = (
    "Task description: denoising a radio signal based on {pred_len} samples with noise; "
)
TASK_PROMPT_IMPUTE: str = (
    "Task description: recovering a missing radio signal based on {pred_len} samples; "
)

# Full hybrid prompt template used in the original repository.
PROMPT_TEMPLATE: str = (
    "<|start_prompt|>Dataset description: {dataset_description}"
    "{task_prompt}"
    "Input statistics: "
    "min value {min_value}, "
    "max value {max_value}, "
    "median value {median_value}, "
    "the trend of input is {trend}, "
    "top 5 lags are : {lags}<|<end_prompt>|>"
)


class RadioLLMConfig(PretrainedConfig):
    """Configuration for :class:`RadioLLMModel`.

    Defaults follow ``repo/RadioLLM/finetune.py`` (GPT2 + classification).
    """

    model_type: str = "radiollm"

    def __init__(
        self,
        seq_len: int = 128,
        n_classes: int = 11,
        enc_in: int = 2,
        d_model: int = 128,
        d_ff: int = 32,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        llm_model: str = "GPT2",
        llm_path: str = "gpt2",
        llm_dim: int = 768,
        llm_layers: int = 6,
        K: int = 7,
        top_k: int = 5,
        num_tokens: int = 1000,
        is_LORA: bool = False,
        pretrained: bool = True,
        use_prompt: bool = True,
        dataset_key: str = "RML2016a",
        dataset: Optional[str] = None,
        dataset_description: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_len: int = seq_len
        self.n_classes: int = n_classes
        self.enc_in: int = enc_in
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.patch_len: int = patch_len
        self.stride: int = stride
        self.llm_model: str = llm_model
        self.llm_path: str = llm_path
        self.llm_dim: int = llm_dim
        self.llm_layers: int = llm_layers
        self.K: int = K
        self.top_k: int = top_k
        self.num_tokens: int = num_tokens
        self.is_LORA: bool = is_LORA
        self.pretrained: bool = pretrained
        self.use_prompt: bool = use_prompt
        # ``dataset`` is the argparse name from ``main.py``; prefer it when set.
        self.dataset_key: str = dataset if dataset is not None else dataset_key
        self.dataset: str = self.dataset_key
        self.dataset_description: str = (
            dataset_description
            if dataset_description is not None
            else DATASET_DESCRIPTIONS.get(self.dataset_key, DEFAULT_DATASET_DESCRIPTION)
        )


class RadioLLMModel(nn.Module):
    """`RadioLLM <https://arxiv.org/abs/2501.17888>`_ classification backbone.

    Input shape: ``[Batch, 2, seq_len]`` (I/Q).
    Output shape: ``[Batch, n_classes]``.
    """

    config_class = RadioLLMConfig

    def __init__(self, config: RadioLLMConfig) -> None:
        super().__init__()
        if config.llm_model.upper() != "GPT2":
            raise ValueError(
                "This port currently supports llm_model='GPT2' "
                f"(got {config.llm_model!r})."
            )

        self.config: RadioLLMConfig = config
        self.seq_len: int = config.seq_len
        self.n_classes: int = config.n_classes
        self.d_ff: int = config.d_ff
        self.d_llm: int = config.llm_dim
        self.patch_len: int = config.patch_len
        self.stride: int = config.stride
        self.top_k: int = config.top_k
        self.K: int = config.K
        self.use_prompt: bool = config.use_prompt
        self.dataset_description: str = config.dataset_description
        self.pred_len: int = config.seq_len

        self.cls_tokens: nn.Parameter = nn.Parameter(torch.zeros(1, 1, self.d_llm))
        nn.init.normal_(self.cls_tokens, std=0.02)

        self._build_llm(config)

        self.high_freq_extract: High_freq_conv_3layer = High_freq_conv_3layer(
            d_model=config.d_model,
            out_channel=self.d_llm,
            patch_len=max(1, self.patch_len // 4),
            stride=1,
            dropout=0.0,
        )
        self.attn_fusion: AttentionFusion = AttentionFusion(self.d_llm)
        self.patch_embedding: PatchEmbedding = PatchEmbedding(
            config.d_model, self.patch_len, self.stride, config.dropout
        )
        self.normalize_layers: Normalize = Normalize(config.enc_in, affine=False)

        self.word_embeddings: torch.Tensor
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size: int = self.word_embeddings.shape[0]
        self.mapping_layer: nn.Linear = nn.Linear(self.vocab_size, config.num_tokens)
        self.reprogramming_layer: ReprogrammingLayer = ReprogrammingLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            self.d_llm,
            attention_dropout=config.dropout,
        )

        self.pool: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.mlp_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.d_llm * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.n_classes),
        )

        if config.is_LORA:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required when is_LORA=True. Install with: pip install peft"
                )
            peft_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            for name, param in self.llm_model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
        else:
            for param in self.llm_model.parameters():
                param.requires_grad = False

    def _build_llm(self, config: RadioLLMConfig) -> None:
        gpt2_config: GPT2Config = GPT2Config.from_pretrained(config.llm_path)
        gpt2_config.num_hidden_layers = config.llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        # Keep embedding dim consistent with llm_dim when using standard GPT2.
        if config.llm_dim != gpt2_config.n_embd:
            gpt2_config.n_embd = config.llm_dim
            gpt2_config.n_inner = 4 * config.llm_dim

        if config.pretrained and config.llm_dim == 768:
            self.llm_model: GPT2Model = GPT2Model.from_pretrained(
                config.llm_path, config=gpt2_config
            )
        else:
            self.llm_model = GPT2Model(config=gpt2_config)

        try:
            self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
                config.llm_path, local_files_only=True
            )
        except EnvironmentError:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                config.llm_path, local_files_only=False
            )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = "[PAD]"

    def calcute_lags(self, x_enc: torch.Tensor) -> torch.Tensor:
        q_fft: torch.Tensor = torch.fft.rfft(
            x_enc.permute(0, 2, 1).contiguous(), dim=-1
        )
        k_fft: torch.Tensor = torch.fft.rfft(
            x_enc.permute(0, 2, 1).contiguous(), dim=-1
        )
        res: torch.Tensor = q_fft * torch.conj(k_fft)
        corr: torch.Tensor = torch.fft.irfft(res, dim=-1)
        mean_value: torch.Tensor = torch.mean(corr, dim=1)
        k: int = min(self.top_k, mean_value.shape[-1])
        _, lags = torch.topk(mean_value, k, dim=-1)
        return lags

    def build_prompts(
        self, x_enc_bn: torch.Tensor, enable_mask: bool = False
    ) -> List[str]:
        """Build hybrid text prompts (synced from ``repo/RadioLLM``).

        Args:
            x_enc_bn: Per-channel tensor of shape ``[B * N, T, 1]``.
            enable_mask: Selects denoise vs imputation task wording.
        """
        min_values: torch.Tensor = torch.min(x_enc_bn, dim=1)[0]
        max_values: torch.Tensor = torch.max(x_enc_bn, dim=1)[0]
        medians: torch.Tensor = torch.median(x_enc_bn, dim=1).values
        lags: torch.Tensor = self.calcute_lags(x_enc_bn)
        trends: torch.Tensor = x_enc_bn.diff(dim=1).sum(dim=1)

        task_prompt: str = (
            TASK_PROMPT_IMPUTE if enable_mask else TASK_PROMPT_DENOISE
        ).format(pred_len=self.pred_len)

        prompts: List[str] = []
        for b in range(x_enc_bn.shape[0]):
            prompts.append(
                PROMPT_TEMPLATE.format(
                    dataset_description=self.dataset_description,
                    task_prompt=task_prompt,
                    min_value=str(min_values[b].tolist()[0]),
                    max_value=str(max_values[b].tolist()[0]),
                    median_value=str(medians[b].tolist()[0]),
                    trend="upward" if trends[b] > 0 else "downward",
                    lags=str(lags[b].tolist()),
                )
            )
        return prompts

    def forward(self, x: torch.Tensor, enable_mask: bool = False) -> torch.Tensor:
        """
        Args:
            x: IQ tensor ``[B, 2, seq_len]``.
            enable_mask: Kept for API parity with the original repo
                (affects task prompt wording only in this port).
        """
        if self.use_prompt:
            return self.classific(x, enable_mask=enable_mask)
        return self.classific_without_prompt(x)

    def classific(self, x_enc: torch.Tensor, enable_mask: bool = False) -> torch.Tensor:
        x_enc_ori: torch.Tensor = x_enc
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, "norm")

        B: int
        T: int
        N: int
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        prompts: List[str] = self.build_prompts(x_enc, enable_mask=enable_mask)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt_ids: torch.Tensor = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids
        prompt_embeddings: torch.Tensor = self.llm_model.get_input_embeddings()(
            prompt_ids.to(x_enc.device)
        )

        source_embeddings: torch.Tensor = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out_sgn, n_vars = self.patch_embedding(x_enc)
        enc_out_sgn_1: torch.Tensor = self.high_freq_extract(x_enc_ori)
        enc_out_sgn_1 = einops.rearrange(
            enc_out_sgn_1, "b (n d) l -> (b n) l d", n=n_vars
        )

        enc_out: torch.Tensor = self.reprogramming_layer(
            enc_out_sgn, source_embeddings, source_embeddings
        )
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)

        prompt_word_embeddings: torch.Tensor = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(
            prompt_embeddings, prompt_word_embeddings, self.K
        )

        cls_tokens: torch.Tensor = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        prompt_length: int = prompt_embeddings.shape[1]
        llama_enc_out: torch.Tensor = torch.cat(
            [prompt_embeddings, enc_out, cls_tokens], dim=1
        )
        dec_out: torch.Tensor = self.llm_model(
            inputs_embeds=llama_enc_out
        ).last_hidden_state
        dec_out = einops.rearrange(dec_out, "(b n) l d -> b l (n d)", n=2)
        dec_out_con: torch.Tensor = dec_out[:, prompt_length:-1, :]
        dec_out_con = self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        return self.mlp_head(dec_out_con)

    def classific_without_prompt(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Ablation path without text prompts (still uses token reprogramming)."""
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, "norm")
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        source_embeddings: torch.Tensor = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )
        cls_tokens: torch.Tensor = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        llama_enc_out: torch.Tensor = torch.cat([enc_out, cls_tokens], dim=1)
        dec_out: torch.Tensor = self.llm_model(
            inputs_embeds=llama_enc_out
        ).last_hidden_state
        dec_out = einops.rearrange(dec_out, "(b n) l d -> b l (n d)", n=n_vars)
        dec_out_con: torch.Tensor = dec_out[:, :-1, :]
        dec_out_con = self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        return self.mlp_head(dec_out_con)
