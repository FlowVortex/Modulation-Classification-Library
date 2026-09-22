import unittest

from utils.dataset import RML2016aDataLoader, RML2016bDataLoader, RML2018aDataLoader


class DataSetConfigs(object):
    """Loading dataset configuration class"""

    def __init__(self, dataset: str, file_path: str, root_path: str) -> None:
        self.batch_size = 128
        self.num_workers = 0
        self.shuffle = True

        self.snr = 0

        self.split_ratio = 0.6

        self.dataset = dataset
        self.file_path = file_path
        self.root_path = root_path


class TestDataset(unittest.TestCase):
    """Test various class methods for loading datasets"""

    def test_load_RML2016a(self) -> None:
        """Test loading RML2016a"""

        configs = DataSetConfigs(
            dataset="RML2016a",
            file_path="/root/autodl-tmp/dataset/RML2016.10a_dict.pkl",
            root_path=None,
        )
        train_loader, val_loader, test_loader = RML2016aDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 128)

        # Check if the proportions of the dataset allocation are correct.
        train_num, val_num, test_num = (
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )
        num_data = train_num + val_num + test_num
        self.assertEqual(train_num / num_data, 0.6)
        self.assertEqual(val_num / num_data, 0.2)
        self.assertEqual(test_num / num_data, 0.2)

    def test_load_RML2016b(self) -> None:
        """Test loading RML2016b"""

        configs = DataSetConfigs(
            dataset="RML2016b", file_path="/root/autodl-tmp/dataset/RML2016.10b.dat", root_path=None
        )
        train_loader, val_loader, test_loader = RML2016bDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 128)

        # Check if the proportions of the dataset allocation are correct.
        train_num, val_num, test_num = (
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )
        num_data = train_num + val_num + test_num
        self.assertEqual(train_num / num_data, 0.6)
        self.assertEqual(val_num / num_data, 0.2)
        self.assertEqual(test_num / num_data, 0.2)

    def test_load_RML2018a(self) -> None:
        """Test loading RML2018a"""

        configs = DataSetConfigs(
            dataset="RML2018a",
            file_path="/root/autodl-tmp/dataset/GOLD_XYZ_OSC.0001_1024.hdf5",
            root_path=None,
        )
        train_loader, val_loader, test_loader = RML2018aDataLoader(configs).load()

        # Obtain data for forward propagation
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # Check if the data format is correct.
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 1024)

        # Check if the proportions of the dataset allocation are correct.
        train_num, val_num, test_num = (
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )
        num_data = train_num + val_num + test_num
        self.assertAlmostEqual(train_num / num_data, 0.6, places=4)
        self.assertAlmostEqual(val_num / num_data, 0.2, places=4)
        self.assertAlmostEqual(test_num / num_data, 0.2, places=4)


if __name__ == "__main__":
    unittest.main()
