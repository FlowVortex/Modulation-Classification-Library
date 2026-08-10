import unittest
import torch

from pathlib import Path

import sys

# Setting up project root for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model import (
    AMCNetConfig,
    AMCNetModel,
    CDATConfig,
    CDATModel,
    CTNetConfig,
    CTNetModel,
    DenseCNNConfig,
    DenseCNNModel,
    DP_DRSNConfig,
    DP_DRSNModel,
    EMC2NetConfig,
    EMC2NetModel,
    InceptionTimeConfig,
    InceptionTimeModel,
    MCformerConfig,
    MCformerModel,
    MCLDNNConfig,
    MCLDNNModel,
    MTAMRConfig,
    MTAMRModel,
    ModernTCNConfig,
    ModernTCNModel,
    PETCGDNNConfig,
    PETCGDNNModel,
)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.batch_size = 4
        # Standard IQ Input: (Batch, Channels, Length)
        cls.common_input = torch.rand((cls.batch_size, 2, 128)).to(cls.device)
        cls.n_classes = 11

    def _run_test(self, model_instance, input_data, expected_shape) -> None:
        model_instance.to(self.device)
        model_instance.eval()
        with torch.no_grad():
            outputs = model_instance(input_data)
        self.assertEqual(outputs.shape, expected_shape)

    def test_all_models(self) -> None:
        test_cases = [
            (AMCNetConfig, AMCNetModel, {"d_model": 36, "n_heads": 2, "d_ff": 512}),
            (CDATConfig, CDATModel, {"d_model": 32, "n_heads": 4}),
            (CTNetConfig, CTNetModel, {"d_model": 64}),
            (DP_DRSNConfig, DP_DRSNModel, {}),
            (EMC2NetConfig, EMC2NetModel, {}),
            (InceptionTimeConfig, InceptionTimeModel, {"d_model": 32}),
            (MCformerConfig, MCformerModel, {"d_model": 64, "n_heads": 8}),
            (MCLDNNConfig, MCLDNNModel, {}),
            (MTAMRConfig, MTAMRModel, {"d_model": 64}),
            (PETCGDNNConfig, PETCGDNNModel, {}),
            (ModernTCNConfig, ModernTCNModel, {}),
            (DenseCNNConfig, DenseCNNModel, {"d_model": 64, "growth_rate": 12}),
        ]

        for config_cls, model_cls, overrides in test_cases:
            name = model_cls.__name__
            with self.subTest(model=name):
                cfg = config_cls(n_classes=self.n_classes, seq_len=128, **overrides)
                model = model_cls(cfg)
                self._run_test(
                    model, self.common_input, (self.batch_size, self.n_classes)
                )


if __name__ == "__main__":
    unittest.main()
