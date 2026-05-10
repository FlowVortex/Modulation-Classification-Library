import unittest
import torch

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Any

import sys

# Setting up project root for imports
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
    n_blocks: int = 6
    batch_size: int = 4

    # AMCNet
    conv_chan_list: Any = None

    # DenseNet
    growth_rate: int = 32
    block_config: Tuple = (6, 12, 24, 16)
    bn_size: int = 4
    reduction: float = 0.5

    # ModernTCN 
    patch_len: int = 8
    stride: int = 4


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.batch_size = 4
        # Standard IQ Input: (Batch, Channels, Length)
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
            (AMCNet.Model, {"d_model": 36, "n_heads": 2, "d_ff": 512}, "AMCNet"),
            (CDAT.Model, {"d_model": 32, "n_heads": 4}, "CDAT"),
            (CTNet.Model, {"d_model": 64}, "CTNet"),
            (DP_DRSN.Model, {"d_model": 63}, "DP_DRSN"),
            (EMC2Net.Model, {}, "EMC2Net"),
            (InceptionTime.Model, {"d_model": 32}, "InceptionTime"),
            (MCformer.MCformer, {"d_model": 64, "n_heads": 8}, "MCformer"),
            (MCLDNN.Model, {}, "MCLDNN"),
            (MTAMR.Model, {"d_model": 64}, "MTAMR"),
            (PETCGDNN.Model, {}, "PETCGDNN"),
            (ModernTCN.Model, {}, "ModernTCN"),
            # DenseCNN integrated: now takes (Batch, 2, Length)
            (DenseCNN.Model, {"d_model": 64, "growth_rate": 12}, "DenseCNN"),
        ]

        for model_fn, overrides, name in test_cases:
            with self.subTest(model=name):
                cfg = ModelConfig(**{**asdict(base_cfg), **overrides})
                model = model_fn(configs=cfg)
                self._run_test(model, self.common_input, (self.batch_size, cfg.n_classes))


if __name__ == "__main__":
    unittest.main()
