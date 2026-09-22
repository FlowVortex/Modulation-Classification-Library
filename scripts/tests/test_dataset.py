import unittest

from utils.dataset import RML2016aDataLoader, RML2016bDataLoader, RML2018aDataLoader

# 预划分好的 HF arrow 数据根目录（其下为 RML2016a/RML2016b/RML2018a，各含 train/val/test）
DATASET_ROOT = "/data/wrz/rml"


class DataSetConfigs(object):
    """Loading dataset configuration class"""

    def __init__(self, dataset: str, root_path: str) -> None:
        self.batch_size = 128
        self.num_workers = 0
        self.shuffle = True

        self.snr = 0
        self.split_ratio = 0.6

        self.dataset = dataset
        self.root_path = root_path
        self.file_path = None
        self.task_name = "AMC"
        self.snr_list = None


class TestDataset(unittest.TestCase):
    """Test various class methods for loading datasets"""

    def test_load_RML2016a(self) -> None:
        """Test loading RML2016a"""

        configs = DataSetConfigs(dataset="RML2016a", root_path=DATASET_ROOT)
        train_loader, val_loader, test_loader = RML2016aDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 128)

        # 数据已预划分为 train/val/test，三个 split 都应非空
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)

    def test_load_RML2016b(self) -> None:
        """Test loading RML2016b"""

        configs = DataSetConfigs(dataset="RML2016b", root_path=DATASET_ROOT)
        train_loader, val_loader, test_loader = RML2016bDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 128)

        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)

    def test_load_RML2018a(self) -> None:
        """Test loading RML2018a"""

        configs = DataSetConfigs(dataset="RML2018a", root_path=DATASET_ROOT)
        train_loader, val_loader, test_loader = RML2018aDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 1024)

        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)


if __name__ == "__main__":
    unittest.main()
