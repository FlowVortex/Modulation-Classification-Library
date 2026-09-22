from typing import Optional, Union, Tuple, List, Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, h5py, gc

class ModulationFineTuningDataset(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, torch.FloatTensor],
        labels: Union[np.ndarray, torch.LongTensor],
    ) -> None:
        super().__init__()
        self.features = features
        self.labels = labels.long()
        self._dataset_length = len(self.labels)

    def __len__(self) -> int:
        return self._dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class BaseDataLoader(object):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.shuffle = configs.shuffle
        self.root_path = configs.root_path
        self.file_path = getattr(configs, 'file_path', None)

        # 获取 SNR 列表：优先使用 snr_list，否则退化到单 snr
        self.target_snr = configs.snr
        snr_list_from_config = getattr(configs, 'snr_list', None)
        if snr_list_from_config is not None:
            self.snr_list = snr_list_from_config
        else:
            self.snr_list = [self.target_snr]

        self.val_batch_size = 128
        self.val_test_split_ratio = 0.4      # 验证+测试共 40%，各 20%
        self.task_name = getattr(configs, 'task_name', 'AMC')
        self.data_ratio = getattr(configs, 'data_ratio', 1.0)

        # 用于存储按 SNR 分离的测试数据，供测试时单独评估
        self._snr_test_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.snr_test_loaders: Dict[int, DataLoader] = {}

    @classmethod
    def load_pkl(cls, file_path: str) -> Dict:
        return pickle.load(open(file_path, "rb"), encoding="iso-8859-1")

    @classmethod
    def load_dat(cls, file_path: str) -> Dict:
        return pickle.load(open(file_path, "rb"), encoding="iso-8859-1")

    @classmethod
    def load_h5py(cls, file_path: str) -> Dict:
        return h5py.File(file_path, "r")

    @staticmethod
    def add_noise(x, std):
        """叠加高斯白噪声，按 std 控制噪声强度"""
        noise = np.random.randn(*x.shape).astype(np.float32) * std
        return (x + noise).astype(np.float32)

    def generate_noise_data(self, shape: Tuple, std: float) -> np.ndarray:
        """生成纯噪声样本（零均值高斯分布），std 为噪声标准差，需与信号功率匹配"""
        return np.random.normal(0, std, size=shape).astype(np.float32)

    def get_data_loader(
        self, train_dataset, val_dataset, test_dataset, batch_size=None, shuffle=None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle if shuffle is not None else self.shuffle,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader, test_loader

    def normalize(self, X_train, X_val, X_test):
        # 使用 float32 避免 StandardScaler 内部 float64 转换造成 2× 内存膨胀
        self.scaler = StandardScaler()
        ns, nc, sl = X_train.shape
        X_train_flat = X_train.reshape(ns, -1)
        self.scaler.fit(X_train_flat)

        def transform(X):
            _ns, _nc, _sl = X.shape
            X_flat = X.reshape(_ns, -1)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.astype(np.float32).reshape(_ns, _nc, _sl)

        return transform(X_train), transform(X_val), transform(X_test)

    def _build_snr_test_loaders(self):
        """将 _snr_test_data 中的各 SNR 测试数据做归一化并构建 DataLoader"""
        self.snr_test_loaders = {}
        for snr, (X_te, y_te) in self._snr_test_data.items():
            ns, nc, sl = X_te.shape
            X_flat = X_te.reshape(ns, -1)
            X_scaled = self.scaler.transform(X_flat).astype(np.float32)
            X_te_norm = X_scaled.reshape(ns, nc, sl)

            ds = ModulationFineTuningDataset(
                torch.FloatTensor(X_te_norm),
                torch.LongTensor(y_te),
            )
            self.snr_test_loaders[snr] = DataLoader(
                ds, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
            )

    def process_labels(self, y: np.ndarray, class_list: List[str]) -> np.ndarray:
        if self.task_name == 'WTC':
            if len(class_list) <= 11:
                wtc_mapping = {0:0, 3:0, 9:0, 7:1, 8:1, 4:2, 5:2, 1:3, 2:3, 10:3, 6:4}
            else:
                wtc_mapping = {
                    0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:1, 7:1, 23:1,
                    8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2, 15:2, 16:2,
                    17:3, 18:3, 19:3, 20:3, 21:3, 22:4
                }
            return np.array([wtc_mapping[int(i)] for i in y])
        return y


    # ---- HF arrow 本地数据加载 ----

    @staticmethod
    def _col_to_np(col):
        """将 arrow 列安全地转成 numpy 数组（兼容固定形状 Array 与 list 两种存储）。"""
        arr = np.asarray(col)
        if arr.dtype == object:
            arr = np.stack([np.asarray(r) for r in col])
        return arr

    def _load_arrow_split(self, split_dir: str):
        """加载本地 HF arrow 格式的单个 split（train/test/val）。"""
        from datasets import load_from_disk

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"未找到 arrow 数据目录: {split_dir}。"
                f"请确认数据位于 {self.root_path} 下且已按 train/test/val 划分。"
            )
        return load_from_disk(split_dir)

    def _arrow_to_arrays(self, ds):
        """将 arrow Dataset 转成 (X, y, z) numpy 数组，X 统一为 (N, 2, L)。"""
        X = self._col_to_np(ds["X"]).astype(np.float32)
        # 统一成 (N, 通道数=2, 序列长度 L)，兼容 (N, L, 2) 的存储方式
        if X.ndim == 3 and X.shape[1] != 2 and X.shape[2] == 2:
            X = np.transpose(X, (0, 2, 1))

        Y = self._col_to_np(ds["Y"])
        y = Y.argmax(axis=1).astype(np.int64) if Y.ndim > 1 else Y.astype(np.int64)

        Z = self._col_to_np(ds["Z"]).reshape(-1).astype(np.int64)
        return X, y, Z

    def _ss_add_noise(self, X: np.ndarray, z: np.ndarray):
        """SS 任务：为每个 SNR 的信号叠加匹配功率的噪声，返回 (X, y, z)。"""
        Xs, ys, zs = [], [], []
        for snr in self.snr_list:
            mask = z == snr
            X_sig = X[mask]
            if len(X_sig) == 0:
                continue
            rms = np.sqrt(np.mean(X_sig ** 2))
            X_noise = self.generate_noise_data(X_sig.shape, std=rms)
            n_sig, n_noise = len(X_sig), len(X_noise)
            Xs.append(np.vstack([X_sig, X_noise]))
            ys.append(np.hstack([np.ones(n_sig), np.zeros(n_noise)]))
            zs.append(np.full(n_sig + n_noise, snr, dtype=np.int64))
        if not Xs:
            nc, sl = X.shape[1], X.shape[2]
            return (
                np.empty((0, nc, sl), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return (
            np.vstack(Xs).astype(np.float32),
            np.hstack(ys).astype(np.int64),
            np.hstack(zs).astype(np.int64),
        )

    def load_arrow(self, dataset_name: str, batch_size=None, shuffle=None):
        """从预划分的 HF arrow 数据（train/test/val）加载并构建 DataLoader。

        与旧版 pkl/dat/hdf5 加载器不同，这里直接读取已经划分好的
        train / test / val 三个文件夹，不再在内部做 train_test_split。
        """
        dataset_dir = os.path.join(self.root_path, dataset_name)

        X_train, y_train, z_train = self._arrow_to_arrays(
            self._load_arrow_split(os.path.join(dataset_dir, "train"))
        )
        X_val, y_val, z_val = self._arrow_to_arrays(
            self._load_arrow_split(os.path.join(dataset_dir, "val"))
        )
        X_test, y_test, z_test = self._arrow_to_arrays(
            self._load_arrow_split(os.path.join(dataset_dir, "test"))
        )

        # 仅保留 snr_list 中指定的 SNR
        snr_arr = np.asarray(self.snr_list)
        tr_mask = np.isin(z_train, snr_arr)
        va_mask = np.isin(z_val, snr_arr)
        te_mask = np.isin(z_test, snr_arr)
        X_train, y_train, z_train = X_train[tr_mask], y_train[tr_mask], z_train[tr_mask]
        X_val, y_val, z_val = X_val[va_mask], y_val[va_mask], z_val[va_mask]
        X_test, y_test, z_test = X_test[te_mask], y_test[te_mask], z_test[te_mask]

        # 任务相关的标签处理
        if self.task_name == 'SS':
            X_train, y_train, z_train = self._ss_add_noise(X_train, z_train)
            X_val, y_val, z_val = self._ss_add_noise(X_val, z_val)
            X_test, y_test, z_test = self._ss_add_noise(X_test, z_test)
        elif self.task_name == 'AD':
            raise ValueError(
                "AD 任务需要数据集中包含 'noise' 类别，但 RML 调制数据中不含噪声类别。"
            )
        elif self.task_name == 'WTC':
            y_train = self.process_labels(y_train, self.class_list)
            y_val = self.process_labels(y_val, self.class_list)
            y_test = self.process_labels(y_test, self.class_list)

        # 按 SNR 分离的测试数据（保存归一化前的原始数据）
        self._snr_test_data = {}
        for snr in self.snr_list:
            mask = z_test == snr
            if mask.any():
                self._snr_test_data[snr] = (X_test[mask], y_test[mask])

        # 归一化（仅用训练集拟合 scaler）
        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        # 构建按 SNR 分离的测试 DataLoader
        self._build_snr_test_loaders()

        train_ds = ModulationFineTuningDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = ModulationFineTuningDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        test_ds = ModulationFineTuningDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        return self.get_data_loader(train_ds, val_ds, test_ds, batch_size, shuffle)


# ================== RML2016a ==================
class RML2016aDataLoader(BaseDataLoader):

    @property
    def class_list(self) -> List[str]:
        return ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

    def load(self, batch_size=None, shuffle=None):
        return self.load_arrow("RML2016a", batch_size, shuffle)


# ================== RML2016b ==================
class RML2016bDataLoader(BaseDataLoader):

    @property
    def class_list(self) -> List[str]:
        return ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

    def load(self, batch_size=None, shuffle=None):
        return self.load_arrow("RML2016b", batch_size, shuffle)


# ================== RML2018a ==================
class RML2018aDataLoader(BaseDataLoader):

    @property
    def class_list(self) -> List[str]:
        return [
            "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
            "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
            "FM", "GMSK", "OQPSK"
        ]

    def load(self, batch_size=None, shuffle=None):
        return self.load_arrow("RML2018a", batch_size, shuffle)


# ================== AD Benchmark Datasets (MSL / PSM / SMAP / SMD) ==================
HUGGINGFACE_REPO = "thuml/Time-Series-Library"


class ADSegDataLoader(BaseDataLoader):
    """AD 异常检测基准数据加载器，支持 MSL / PSM / SMAP / SMD。

    数据格式:
      - MSL/SMAP/SMD: {name}_train.npy, {name}_test.npy, {name}_test_label.npy
      - PSM: train.csv, test.csv, test_label.csv
    所有数据为 2D (timesteps, features)，内部通过滑动窗口切分为样本。
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.win_size = configs.seq_len
        self.step = getattr(configs, 'step', 1)

    @property
    def class_list(self) -> List[str]:
        return ["Normal", "Anomaly"]

    # ---- 底层数据加载 (本地优先，回退 HuggingFace) ----

    def _load_npy_dataset(self, data_dir: str, dataset_name: str):
        """加载 MSL / SMAP / SMD 的 .npy 文件"""
        hf_base = f"{dataset_name}/{dataset_name}"
        train_f = os.path.join(data_dir, f"{dataset_name}_train.npy")
        test_f  = os.path.join(data_dir, f"{dataset_name}_test.npy")
        label_f = os.path.join(data_dir, f"{dataset_name}_test_label.npy")

        if all(os.path.exists(p) for p in [train_f, test_f, label_f]):
            train  = np.load(train_f)
            test   = np.load(test_f)
            labels = np.load(label_f)
        else:
            from huggingface_hub import hf_hub_download
            train  = np.load(hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=f"{hf_base}_train.npy", repo_type="dataset"))
            test   = np.load(hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=f"{hf_base}_test.npy",  repo_type="dataset"))
            labels = np.load(hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=f"{hf_base}_test_label.npy", repo_type="dataset"))
        return train, test, labels

    def _load_psm(self, data_dir: str):
        """加载 PSM 的 CSV 文件"""
        train_f = os.path.join(data_dir, "train.csv")
        test_f  = os.path.join(data_dir, "test.csv")
        label_f = os.path.join(data_dir, "test_label.csv")

        if all(os.path.exists(p) for p in [train_f, test_f, label_f]):
            train_df      = pd.read_csv(train_f)
            test_df       = pd.read_csv(test_f)
            test_label_df = pd.read_csv(label_f)
        else:
            from datasets import load_dataset
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        train  = np.nan_to_num(train_df.values[:, 1:])
        test   = np.nan_to_num(test_df.values[:, 1:])
        labels = test_label_df.values[:, 1:].astype(float)
        return train, test, labels

    # ---- 滑动窗口构建 ----

    @staticmethod
    def _build_windows(data: np.ndarray, labels: np.ndarray,
                       win_size: int, step: int):
        """将 (timesteps, features) 切分为滑动窗口。

        Returns:
            X: (n_windows, features, win_size)  兼容模型 (C, L) 输入
            y: (n_windows,)  窗口级标签: 窗口内任一点异常 → 1
        """
        windows, w_labels = [], []
        for i in range(0, len(data) - win_size + 1, step):
            windows.append(data[i:i + win_size])
            w_labels.append(int(np.any(labels[i:i + win_size] > 0)))
        if not windows:
            return (np.empty((0, data.shape[1], win_size), dtype=np.float32),
                    np.array([], dtype=int))
        # (N, win, feat) → (N, feat, win)
        X = np.array(windows, dtype=np.float32).transpose(0, 2, 1)
        return X, np.array(w_labels, dtype=int)

    # ---- 主入口 ----

    def load(self, batch_size=None, shuffle=None):
        dataset_name = self.configs.dataset  # "MSL" / "PSM" / "SMAP" / "SMD"
        data_dir = os.path.join(self.root_path, dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        # 1. 加载原始数据
        if dataset_name == "PSM":
            train_data, test_data, test_labels_raw = self._load_psm(data_dir)
        else:
            train_data, test_data, test_labels_raw = self._load_npy_dataset(data_dir, dataset_name)

        # 2. 标准化 (仅用训练集 fit)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        # 3. 同步 enc_in (模型构建时使用)
        self.configs.enc_in = train_data.shape[1]

        # 4. 滑动窗口切分
        X_tr_all, y_tr_all = self._build_windows(
            train_data, np.zeros(len(train_data)), self.win_size, self.step)
        X_te, y_te = self._build_windows(
            test_data, test_labels_raw, self.win_size, self.step)

        # 5. 训练集 80/20 拆分为 train / val
        split = int(len(X_tr_all) * 0.8)
        X_tr, y_tr = X_tr_all[:split], y_tr_all[:split]
        X_va, y_va = X_tr_all[split:], y_tr_all[split:]

        # 6. 构建 DataLoader
        train_ds = ModulationFineTuningDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_ds   = ModulationFineTuningDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va))
        test_ds  = ModulationFineTuningDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te))
        return self.get_data_loader(train_ds, val_ds, test_ds, batch_size, shuffle)
