from typing import Optional, Union, Tuple, List, Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, h5py

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
        self.file_path = configs.file_path

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
        self.scaler = StandardScaler()
        ns, nc, sl = X_train.shape
        X_train_flat = X_train.reshape(ns, -1)
        self.scaler.fit(X_train_flat)

        def transform(X):
            _ns, _nc, _sl = X.shape
            X_flat = X.reshape(_ns, -1)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(_ns, _nc, _sl)

        return transform(X_train), transform(X_val), transform(X_test)

    def _build_snr_test_loaders(self):
        """将 _snr_test_data 中的各 SNR 测试数据做归一化并构建 DataLoader"""
        self.snr_test_loaders = {}
        for snr, (X_te, y_te) in self._snr_test_data.items():
            ns, nc, sl = X_te.shape
            X_flat = X_te.reshape(ns, -1)
            X_scaled = self.scaler.transform(X_flat)
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


# ================== RML2016a ==================
class RML2016aDataLoader(BaseDataLoader):
    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        return ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

    def load(self, batch_size=None, shuffle=None):
        data_dict = self.load_pkl(self.file_path)
        all_mods = sorted(list(set([k[0] for k in data_dict.keys()])))

        X_tr_all, y_tr_all = [], []
        X_va_all, y_va_all = [], []
        X_te_all, y_te_all = [], []

        if self.task_name == 'AD':
            noise_mods = [m for m in all_mods if 'noise' in m.lower()]
            if not noise_mods:
                raise ValueError("AD任务需要数据集中包含 'noise' 类别，但未找到。")

            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                for mod in all_mods:
                    X = data_dict[(mod, snr)]
                    y = np.ones(X.shape[0]) if 'noise' in mod.lower() else np.zeros(X.shape[0])
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=self.val_test_split_ratio, stratify=y)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        elif self.task_name == 'SS':
            # 先收集所有SNR下的非噪声信号，再统一生成噪声、划分
            X_signals_all = []
            for snr in self.snr_list:
                for mod in all_mods:
                    if 'noise' not in mod.lower():
                        X_signals_all.append(data_dict[(mod, snr)])
            X_signals = np.vstack(X_signals_all)
            # 生成匹配信号 RMS 功率的纯噪声（避免模型靠功率差异作弊）
            signal_rms = np.sqrt(np.mean(X_signals ** 2))
            X_noise = self.generate_noise_data(X_signals.shape, std=signal_rms)
            y_signals = np.ones(len(X_signals))
            y_noise = np.zeros(len(X_noise))

            X_all = np.vstack([X_signals, X_noise])
            y_all = np.hstack([y_signals, y_noise])
            X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=self.val_test_split_ratio, stratify=y_all)
            X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)

            # 跳过累积循环，直接赋值
            X_tr_all.append(X_train); y_tr_all.append(y_train)
            X_va_all.append(X_val); y_va_all.append(y_val)
            X_te_all.append(X_test); y_te_all.append(y_test)

        else:  # AMC / WTC
            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                for idx, mod in enumerate(self.class_list):
                    X = data_dict[(mod, snr)]
                    y = np.ones(X.shape[0]) * idx
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=self.val_test_split_ratio, stratify=y)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        X_train = np.vstack(X_tr_all)
        y_train = np.hstack(y_tr_all).astype(int)
        X_val = np.vstack(X_va_all)
        y_val = np.hstack(y_va_all).astype(int)
        X_test = np.vstack(X_te_all)
        y_test = np.hstack(y_te_all).astype(int)

        if self.task_name == 'WTC':
            y_train = self.process_labels(y_train, self.class_list)
            y_val = self.process_labels(y_val, self.class_list)
            y_test = self.process_labels(y_test, self.class_list)
            for snr in list(self._snr_test_data.keys()):
                X_s, y_s = self._snr_test_data[snr]
                self._snr_test_data[snr] = (X_s, self.process_labels(y_s, self.class_list))

        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        # 构建按 SNR 分离的测试 DataLoader
        self._build_snr_test_loaders()

        train_ds = ModulationFineTuningDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = ModulationFineTuningDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = ModulationFineTuningDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        return self.get_data_loader(train_ds, val_ds, test_ds, batch_size, shuffle)


# ================== RML2016b ==================
class RML2016bDataLoader(BaseDataLoader):
    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        return ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

    def load(self, batch_size=None, shuffle=None):
        data_dict = self.load_dat(self.file_path)
        all_mods = sorted(list(set([k[0] for k in data_dict.keys()])))

        X_tr_all, y_tr_all = [], []
        X_va_all, y_va_all = [], []
        X_te_all, y_te_all = [], []

        if self.task_name == 'AD':
            noise_mods = [m for m in all_mods if 'noise' in m.lower()]
            if not noise_mods:
                raise ValueError("AD任务需要数据集中包含 'noise' 类别，但未找到。")

            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                for mod in all_mods:
                    X = data_dict[(mod, snr)]
                    y = np.ones(X.shape[0]) if 'noise' in mod.lower() else np.zeros(X.shape[0])
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=self.val_test_split_ratio, stratify=y)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        elif self.task_name == 'SS':
            X_signals_all = []
            for snr in self.snr_list:
                for mod in all_mods:
                    if 'noise' not in mod.lower():
                        X_signals_all.append(data_dict[(mod, snr)])
            X_signals = np.vstack(X_signals_all)
            # 生成匹配信号 RMS 功率的纯噪声（避免模型靠功率差异作弊）
            signal_rms = np.sqrt(np.mean(X_signals ** 2))
            X_noise = self.generate_noise_data(X_signals.shape, std=signal_rms)
            y_signals = np.ones(len(X_signals))
            y_noise = np.zeros(len(X_noise))

            X_all = np.vstack([X_signals, X_noise])
            y_all = np.hstack([y_signals, y_noise])
            X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=self.val_test_split_ratio, stratify=y_all)
            X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
            X_tr_all.append(X_train); y_tr_all.append(y_train)
            X_va_all.append(X_val); y_va_all.append(y_val)
            X_te_all.append(X_test); y_te_all.append(y_test)

        else:
            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                for idx, mod in enumerate(self.class_list):
                    X = data_dict[(mod, snr)]
                    y = np.ones(X.shape[0]) * idx
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=self.val_test_split_ratio, stratify=y)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        X_train = np.vstack(X_tr_all)
        y_train = np.hstack(y_tr_all).astype(int)
        X_val = np.vstack(X_va_all)
        y_val = np.hstack(y_va_all).astype(int)
        X_test = np.vstack(X_te_all)
        y_test = np.hstack(y_te_all).astype(int)

        if self.task_name == 'WTC':
            y_train = self.process_labels(y_train, self.class_list)
            y_val = self.process_labels(y_val, self.class_list)
            y_test = self.process_labels(y_test, self.class_list)
            for snr in list(self._snr_test_data.keys()):
                X_s, y_s = self._snr_test_data[snr]
                self._snr_test_data[snr] = (X_s, self.process_labels(y_s, self.class_list))

        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        # 构建按 SNR 分离的测试 DataLoader
        self._build_snr_test_loaders()

        train_ds = ModulationFineTuningDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = ModulationFineTuningDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = ModulationFineTuningDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        return self.get_data_loader(train_ds, val_ds, test_ds, batch_size, shuffle)


# ================== RML2018a ==================
class RML2018aDataLoader(BaseDataLoader):
    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        return [
            "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
            "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
            "FM", "GMSK", "OQPSK"
        ]

    def load(self, batch_size=None, shuffle=None):
        data = self.load_h5py(self.file_path)
        X_all_raw = data["X"]
        y_all_raw = np.argmax(data["Y"], axis=1)
        z_all = np.array(data["Z"]).flatten()

        X_tr_all, y_tr_all = [], []
        X_va_all, y_va_all = [], []
        X_te_all, y_te_all = [], []

        if self.task_name == 'AD':
            # 查找噪声类标签
            noise_label = None
            for lbl, name in enumerate(self.class_list):
                if 'noise' in name.lower():
                    noise_label = lbl
                    break
            if noise_label is None:
                raise ValueError("AD任务需要数据集中包含 'noise' 类别，但在 RML2018a 中未找到。")

            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                idx_snr = np.where(z_all == snr)[0]
                if len(idx_snr) == 0:
                    continue
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                X_snr = np.transpose(X_all_raw[idx_snr], (0, 2, 1))
                y_snr = y_all_raw[idx_snr]
                unique_mods = np.unique(y_snr)
                for lbl in unique_mods:
                    idx_mod = np.where(y_snr == lbl)[0]
                    X_mod = X_snr[idx_mod]
                    y_mod = np.ones(len(idx_mod)) if lbl == noise_label else np.zeros(len(idx_mod))
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_mod, y_mod, test_size=self.val_test_split_ratio, stratify=y_mod)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if snr in _snr_X_te and _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        elif self.task_name == 'SS':
            X_signals_all = []
            for snr in self.snr_list:
                idx_snr = np.where(z_all == snr)[0]
                if len(idx_snr) == 0:
                    continue
                X_snr = np.transpose(X_all_raw[idx_snr], (0, 2, 1))
                y_snr = y_all_raw[idx_snr]
                for lbl in np.unique(y_snr):
                    if 'noise' in self.class_list[lbl].lower():
                        continue
                    X_signals_all.append(X_snr[y_snr == lbl])
            if not X_signals_all:
                raise RuntimeError("未找到任何非噪声信号，请检查 SNR 设置或数据集。")
            X_signals = np.vstack(X_signals_all)
            # 生成匹配信号 RMS 功率的纯噪声（避免模型靠功率差异作弊）
            signal_rms = np.sqrt(np.mean(X_signals ** 2))
            X_noise = self.generate_noise_data(X_signals.shape, std=signal_rms)
            y_signals = np.ones(len(X_signals))
            y_noise = np.zeros(len(X_noise))

            X_all = np.vstack([X_signals, X_noise])
            y_all = np.hstack([y_signals, y_noise])
            X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=self.val_test_split_ratio, stratify=y_all)
            X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
            X_tr_all.append(X_train); y_tr_all.append(y_train)
            X_va_all.append(X_val); y_va_all.append(y_val)
            X_te_all.append(X_test); y_te_all.append(y_test)

        else:  # AMC / WTC
            _snr_X_te, _snr_y_te = {}, {}
            for snr in self.snr_list:
                idx_snr = np.where(z_all == snr)[0]
                if len(idx_snr) == 0:
                    continue
                _snr_X_te[snr], _snr_y_te[snr] = [], []
                X_snr = np.transpose(X_all_raw[idx_snr], (0, 2, 1))
                y_snr = y_all_raw[idx_snr]
                for idx, mod in enumerate(self.class_list):
                    idx_mod = np.where(y_snr == idx)[0]
                    if len(idx_mod) == 0:
                        continue
                    X_mod = X_snr[idx_mod]
                    y_mod = np.ones(len(idx_mod)) * idx
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_mod, y_mod, test_size=self.val_test_split_ratio, stratify=y_mod)
                    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)
                    X_tr_all.append(X_tr); y_tr_all.append(y_tr)
                    X_va_all.append(X_va); y_va_all.append(y_va)
                    X_te_all.append(X_te); y_te_all.append(y_te)
                    _snr_X_te[snr].append(X_te); _snr_y_te[snr].append(y_te)
            for snr in self.snr_list:
                if snr in _snr_X_te and _snr_X_te[snr]:
                    self._snr_test_data[snr] = (np.vstack(_snr_X_te[snr]), np.hstack(_snr_y_te[snr]).astype(int))

        X_train = np.vstack(X_tr_all)
        y_train = np.hstack(y_tr_all).astype(int)
        X_val = np.vstack(X_va_all)
        y_val = np.hstack(y_va_all).astype(int)
        X_test = np.vstack(X_te_all)
        y_test = np.hstack(y_te_all).astype(int)

        if self.task_name == 'WTC':
            y_train = self.process_labels(y_train, self.class_list)
            y_val = self.process_labels(y_val, self.class_list)
            y_test = self.process_labels(y_test, self.class_list)
            for snr in list(self._snr_test_data.keys()):
                X_s, y_s = self._snr_test_data[snr]
                self._snr_test_data[snr] = (X_s, self.process_labels(y_s, self.class_list))

        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        # 构建按 SNR 分离的测试 DataLoader
        self._build_snr_test_loaders()

        train_ds = ModulationFineTuningDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = ModulationFineTuningDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = ModulationFineTuningDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        return self.get_data_loader(train_ds, val_ds, test_ds, batch_size, shuffle)

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
