import unittest
import torch

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Any

import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model import (
    AMCNet,
    CDAT,
    CTNet,
    DenseCNN,
    DP_DRSN,
    EMC2Net,
    InceptionTime,
    MCformer,
    MCLDNN,
    MTAMR,
    PETCGDNN,
    ModernTCN,
)


@dataclass
class ModelConfig:
    seq_len: int = 128
    n_classes: int = 11
    input_channels: int = 2

    d_model: int = 64
    d_ff: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # Other
    decimation_factor: int = 8

    # InceptionTime
    n_filters: int = 32
    kernel_sizes: List[int] = field(default_factory=lambda: [9, 19, 39])
    bottleneck_channels: int = 32
    n_blocks: int = 6
    activation: str = "relu"
    bias: bool = False
    use_global_avg_pool: bool = True
    max_pool_size: int = 1
    use_residual: bool = True
    batch_size: int = 4

    # AMCNet
    conv_chan_list: Any = None

    # DenseNet
    growth_rate: int = 32
    block_config: Tuple = (6, 12, 24, 16)
    bn_size: int = 4
    reduction: float = 0.5

    # ModernTCN 
    dims: List[int] = field(default_factory=lambda: [64, 128]) 
    num_blocks: List[int] = field(default_factory=lambda: [2, 2])
    large_size: List[int] = field(default_factory=lambda: [31, 29])
    small_size: List[int] = field(default_factory=lambda: [5, 5])
    patch_size: int = 8
    patch_stride: int = 4
    downsample_ratio: int = 2
    ffn_ratio: int = 4
    class_dropout: float = 0.1
    revin: bool = True


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.batch_size = 4
        # (Batch, Channels, Length)
        cls.common_input = torch.rand((cls.batch_size, 2, 128)).to(cls.device)

    def _run_test(self, model_instance, input_data, expected_shape) -> None:
        model_instance.to(self.device)
        model_instance.eval()
        with torch.no_grad():
            outputs = model_instance(input_data)
        self.assertEqual(outputs.shape, expected_shape)

    def test_all_models(self) -> None:
        base_cfg = ModelConfig()

        test_cases = [
            (AMCNet.model, {"d_model": 36, "n_heads": 2, "d_ff": 512}, "AMCNet"),
            (CDAT.model, {"d_model": 32, "n_heads": 4}, "CDAT"),
            (CTNet.model, {"d_model": 64}, "CTNet"),
            (DP_DRSN.model, {"d_model": 63}, "DP_DRSN"),
            (EMC2Net.model, {}, "EMC2Net"),
            (InceptionTime.model, {}, "InceptionTime"),
            (MCformer.MCformer, {"d_model": 64, "n_heads": 8}, "MCformer"),
            (MTAMR.model, {"d_model": 64}, "MTAMR"),
            (PETCGDNN.model, {}, "PETCGDNN"),
            (ModernTCN.model, {"dims": [64, 128], "num_blocks": [1, 1], }, "ModernTCN"),
        ]

        for model_fn, overrides, name in test_cases:
            with self.subTest(model=name):
                cfg = ModelConfig(**{**asdict(base_cfg), **overrides})
                model = model_fn(configs=cfg)
                self._run_test(model, self.common_input, (4, 11))

    def test_DenseCNN(self) -> None:
        """DenseCNN"""
        cfg = ModelConfig(n_classes=1000)
        model = DenseCNN.model(configs=cfg)
        img_input = torch.rand((2, 3, 224, 224)).to(self.device)
        self._run_test(model, img_input, (2, 1000))

    def test_MCLDNN(self) -> None:
        """MCLDNN"""
        model = MCLDNN.Model(num_classes=11)
        x = self.common_input.unsqueeze(1)  # (batch, 1, 2, L)
        self._run_test(model, x, (4, 11))


if __name__ == "__main__":
    unittest.main()
