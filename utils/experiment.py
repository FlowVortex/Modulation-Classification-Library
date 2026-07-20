from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Dict

import time
import sys
import os
from os import path

from accelerate import Accelerator
from colorama import Fore, Style
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer

from sklearn.metrics import confusion_matrix

from model import MCformer, AMCNet, CDAT, CTNet, DenseCNN, DP_DRSN, EMC2Net, InceptionTime, MCLDNN, MTAMR, PETCGDNN, ModernTCN, ResNet, Conv_AE

from utils.dataset import (
    RML2016aDataLoader,
    RML2016bDataLoader,
    RML2018aDataLoader,
    ADSegDataLoader,
)
from utils.tools import (
    get_loss_fn,
    EarlyStopping,
    OptimInterface,
    logging_results,
    print_configs,
    get_confusion_matrix,
    plot_loss_cruve,
    plot_accuracy_curve,
    plot_confusion_matrix,
)


class BaseExperiment(ABC):
    """Base class for experiments."""

    def __init__(self, configs, accelerator: Accelerator, setting: str) -> None:
        super().__init__()
        self.configs = configs
        self.accelerator = accelerator
        self.setting = setting

        self.model_name = configs.model
        self.dataset = configs.dataset
        self.snr = configs.snr
        self.mode = configs.mode
        self.batch_size = configs.batch_size
        self.shuffle = configs.shuffle
        self.checkpoint_dir = configs.checkpoint
        self.checkpoint_path = path.join(self.checkpoint_dir, setting)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.n_classes = None
        self.class_list = None
        self.file_path = configs.file_path
        self.root_path = configs.root_path
        self.patience = configs.patience

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        return {
            "MCformer": MCformer, "AMCNet": AMCNet, "CDAT": CDAT, "CTNet": CTNet,
            "DenseCNN": DenseCNN, "DP_DRSN": DP_DRSN, "EMC2Net": EMC2Net,
            "InceptionTime": InceptionTime, "MCLDNN": MCLDNN, "MTAMR": MTAMR,
            "PETCGDNN": PETCGDNN, "ModernTCN": ModernTCN, "ResNet":ResNet, "Conv_AE":Conv_AE
        }

    def build_model(self, name: str = "SpectrumTime") -> nn.Module:
        assert name in self.model_dict, f"Model {name} is not supported."
        self.accelerator.print(f"Building the model: {name}", end=" -> ")
        model = self.model_dict[name].Model(self.configs)
        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)
        return model

    @property
    def dataset_dict(self) -> Dict[str, Any]:
        return {
            "RML2016a": RML2016aDataLoader,
            "RML2016b": RML2016bDataLoader,
            "RML2018a": RML2018aDataLoader,
            "MSL": ADSegDataLoader,
            "PSM": ADSegDataLoader,
            "SMAP": ADSegDataLoader,
            "SMD": ADSegDataLoader,
        }

    def load_data(self) -> Tuple[Any, Any, Any]:
        self.data_interface = self.dataset_dict[self.dataset](configs=self.configs)

        # 记录原始类别名
        self.data_interface.task_name = self.configs.task_name
        self.class_list = self.data_interface.class_list
        
        # 动态调整类别数
        if self.configs.task_name == 'WTC':
            # 无论 2016 还是 2018，按照上面的字典映射后都是 5 类
            self.n_classes = 5
            self.class_list = ["Amplitude", "Phase", "High-Order", "Analog", "IoT/Special"]
        elif self.configs.task_name == 'SS':
            self.n_classes = 2
            self.class_list = ["Noise", "Signal"]
        elif self.configs.task_name == 'AD':
            # AD任务虽然测试时是2类，但模型是做重建，通常输出维度等于输入维度，
            # 这里的 n_classes 设为 1 或 保持原样取决于你的模型 Head 逻辑。
            self.n_classes = len(self.class_list) 
        else:
            self.n_classes = len(self.class_list)

        # 反馈给全局配置
        self.configs.n_classes = self.n_classes

        if self.configs.task_name == 'AMC': self.configs.n_classes_amc = self.n_classes
        elif self.configs.task_name == 'WTC': self.configs.n_classes_wtc = self.n_classes
        elif self.configs.task_name == 'SS': self.configs.n_classes_ss = self.n_classes
        
        return self.data_interface.load()

    def _load_optimizer(self):
        return self.optim.load_optimizer(parameters=[p for p in self.model.parameters() if p.requires_grad])

    def _load_scheduler(self, optimizer: Optimizer, loader_len: int):
        return self.optim.load_scheduler(optimizer, loader_len)

    def print_start_message(self, time_now: str) -> None:
        print_configs(
            accelerator=self.accelerator, time_now=time_now,
            config={
                "seq_len": self.configs.seq_len, "epochs": self.configs.num_epochs,
                "batch_size": self.batch_size, "learning_rate": self.configs.learning_rate,
                "optimizer": self.configs.optimizer, "scheduler": self.configs.scheduler,
                "criterion": self.configs.criterion,
            },
            experiment_name=f"Task: {self.configs.task_name}",
            model_name=self.model_name, dataset=self.configs.dataset,
            mode=self.mode, print_separator=True,
        )

    def save_results(self, train_loss, train_acc, val_loss, val_acc, predictions, targets, accuracy, confusion_matrix, time_mean) -> None:
        results_path = self.checkpoint_path + "/results.pth"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(obj={
                "train_loss": torch.tensor(train_loss), "train_acc": torch.tensor(train_acc),
                "val_loss": torch.tensor(val_loss), "val_acc": torch.tensor(val_acc),
                "predictions": predictions, "targets": targets,
                "accuracy": torch.tensor(accuracy), "confusion_matrix": confusion_matrix,
                "time_mean": torch.tensor(time_mean),
            }, f=results_path, safe_serialization=False)
        self.accelerator.print("Test results saved to " + results_path)

    def logging(self, time_now: str, accuracy: float, time_mean: Union[float, np.ndarray]) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            messages = {
                "Timestamp": time_now, "Dataset": self.configs.dataset, "Model": self.model_name,
                "SNR": self.configs.snr, "Accuracy": accuracy, "Time": time_mean,
                "split_ratio": self.configs.split_ratio, "seq_len": self.configs.seq_len,
                "n_classes": self.configs.n_classes, "criterion": self.configs.criterion,
                "optimizer": self.configs.optimizer, "learning_rate": self.configs.learning_rate,
                "batch_size": self.configs.batch_size, "setting": self.setting,
                "seed": self.configs.seed, "path": self.checkpoint_path,
            }
            logging_results(accelerator=self.accelerator, logging_path="./results.csv",
                            headers=["Timestamp", "Dataset", "Model", "SNR", "Accuracy", "Time", "split_ratio", "seq_len", "n_classes", "criterion", "optimizer", "learning_rate", "batch_size", "setting", "seed", "path"],
                            messages=messages)


# =============================================================================
# 1. AMC 任务实验类 (Automatic Modulation Classification)
# =============================================================================
class AMCExperiment(BaseExperiment):
    def __init__(self, configs, accelerator: Accelerator, setting: str, time_now: str) -> None:
        super().__init__(configs, accelerator, setting)
        if not hasattr(self.configs, 'task_name') or self.configs.task_name is None:
            self.configs.task_name = 'AMC'
        self.print_start_message(time_now)
        self.time_now = time_now
        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.model = self.build_model(name=configs.model)
        self.optim = OptimInterface(configs=configs, accelerator=accelerator)
        self.optimizer = self._load_optimizer()
        self.scheduler = self._load_scheduler(self.optimizer, len(self.train_loader))
        self.criterion = get_loss_fn(configs.criterion)

    def train(self, epoch: int):
        self.model.train()
        num_samples, total_loss, total_acc = 0, 0, 0
        data_loader = tqdm(self.train_loader, file=sys.stdout)
        for step, (batch_x, batch_y) in enumerate(data_loader, 1):
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            num_samples += batch_y.size(0)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_acc += torch.eq(predicted, batch_y).sum().item()
            data_loader.desc = f"[AMC Train {epoch}] Loss: {round(total_loss/step, 4)}, Acc: {round(total_acc/num_samples, 4)}"
        return total_loss / step, total_acc / num_samples

    def val(self, epoch: int):
        self.model.eval()
        num_samples, total_loss, total_acc = 0, 0, 0
        data_loader = tqdm(self.val_loader, file=sys.stdout)
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(data_loader, 1):
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                num_samples += batch_y.size(0)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total_acc += torch.eq(predicted, batch_y).sum().item()
                data_loader.desc = f"[AMC Val {epoch}] Loss: {round(total_loss/step, 4)}, Acc: {round(total_acc/num_samples, 4)}"
        return total_loss / step, total_acc / num_samples

    def test(self):
        self.accelerator.load_state(self.checkpoint_path)
        self.model.eval()
        preds, targets, times = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader):
                start = time.time()
                outputs = self.model(batch_x)
                times.append(time.time() - start)
                preds.append(outputs)
                targets.append(batch_y)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        accuracy = torch.eq(torch.max(preds, 1)[1], targets).sum().item() / targets.size(0)

        # ===== 按 SNR 分别测试 =====
        snr_accuracies = {}
        if hasattr(self, 'data_interface') and self.data_interface.snr_test_loaders:
            for snr, snr_loader in self.data_interface.snr_test_loaders.items():
                snr_preds, snr_targets = [], []
                with torch.no_grad():
                    for batch_x, batch_y in snr_loader:
                        batch_x = batch_x.to(self.accelerator.device)
                        batch_y = batch_y.to(self.accelerator.device)
                        outputs = self.model(batch_x)
                        snr_preds.append(outputs)
                        snr_targets.append(batch_y)
                snr_preds = torch.cat(snr_preds, dim=0)
                snr_targets = torch.cat(snr_targets, dim=0)
                snr_acc = torch.eq(torch.max(snr_preds, 1)[1], snr_targets).sum().item() / snr_targets.size(0)
                snr_accuracies[snr] = snr_acc
                self.accelerator.print(f"  -> SNR {snr:3d} dB: Accuracy = {snr_acc:.4f}")
        # =============================

        return accuracy, preds, targets, np.mean(times), snr_accuracies

    def run(self):
        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.patience, verbose=True, delta=self.configs.delta)
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader)
        
        t_loss, t_acc, v_loss, v_acc = [], [], [], []
        for epoch in range(self.configs.num_epochs):
            tr_l, tr_a = self.train(epoch + 1)
            val_l, val_a = self.val(epoch + 1)
            t_loss.append(tr_l); t_acc.append(tr_a); v_loss.append(val_l); v_acc.append(val_a)
            early_stopping(val_a, self.checkpoint_path)
            if early_stopping.early_stop: break

        acc, preds, targets, t_mean, snr_accs = self.test()
        conf_mat = get_confusion_matrix(torch.max(preds, 1)[1], targets, self.n_classes)
        self.save_results(t_loss, t_acc, v_loss, v_acc, preds, targets, acc, conf_mat, t_mean)
        if self.accelerator.is_main_process:
            # plot_loss_cruve(np.array(t_loss), v_loss, self.checkpoint_path)
            plot_accuracy_curve(np.array(t_acc), v_acc, self.checkpoint_path)
            plot_confusion_matrix(np.array(conf_mat), self.checkpoint_path)
        self.logging(self.time_now, acc, t_mean)


# =============================================================================
# 2. WTC 任务实验类 (Wireless Technology Classification) 逻辑同 AMC
# =============================================================================
class WTCExperiment(AMCExperiment):
    def __init__(self, configs, accelerator: Accelerator, setting: str, time_now: str) -> None:
        configs.task_name = 'WTC'
        super().__init__(configs, accelerator, setting, time_now)
        if configs.dataset == 'RML2016a':
            # 定义权重张量
            weights = torch.tensor([1.0/3, 1.0/2, 1.0/2, 1.0/3, 1.0/1], dtype=torch.float)
            
            # 必须把权重搬运到模型所在的设备上 (GPU/CPU)
            weights = weights.to(self.accelerator.device)
            
            # 覆盖原本的 criterion
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            
            self.accelerator.print(f"Successfully updated criterion with WTC weights: {weights.tolist()}")


# =============================================================================
# 3. SS 任务实验类 (Spectrum Sensing) 逻辑同 AMC
# =============================================================================
class SSExperiment(AMCExperiment):
    def __init__(self, configs, accelerator: Accelerator, setting: str, time_now: str) -> None:
        configs.task_name = 'SS'
        super().__init__(configs, accelerator, setting, time_now)


# =============================================================================
# 4. AD 任务实验类 (Anomaly Detection) - 基于重建
# =============================================================================
class ADExperiment(BaseExperiment):
    def __init__(self, configs, accelerator: Accelerator, setting: str, time_now: str) -> None:
        super().__init__(configs, accelerator, setting)
        self.configs.task_name = 'AD'
        self.print_start_message(time_now)
        self.time_now = time_now
        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.model = self.build_model(name=configs.model)
        self.optim = OptimInterface(configs=configs, accelerator=accelerator)
        self.optimizer = self._load_optimizer()
        self.scheduler = self._load_scheduler(self.optimizer, len(self.train_loader))
        self.criterion = nn.MSELoss() # AD 任务固定为 MSE

    def train(self, epoch: int):
        self.model.train()
        total_loss = 0
        data_loader = tqdm(self.train_loader, file=sys.stdout)
        for step, (batch_x, _) in enumerate(data_loader, 1):
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_x) # 重建输入
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            data_loader.desc = f"[AD Train {epoch}] Loss: {round(total_loss/step, 6)}"
        return total_loss / step, 0.0

    def val(self, epoch: int):
        self.model.eval()
        total_loss = 0
        data_loader = tqdm(self.val_loader, file=sys.stdout)
        with torch.no_grad():
            for step, (batch_x, _) in enumerate(data_loader, 1):
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                total_loss += loss.item()
                data_loader.desc = f"[AD Val {epoch}] Loss: {round(total_loss/step, 6)}"
        return total_loss / step, 0.0

    def test(self):
        self.accelerator.load_state(self.checkpoint_path)
        self.model.eval()
        mse_scores, targets, times = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader):
                start = time.time()
                outputs = self.model(batch_x)
                times.append(time.time() - start)
                # 计算每个样本的 MSE
                mse = torch.mean((outputs - batch_x)**2, dim=(1, 2))
                mse_scores.append(mse)
                targets.append(batch_y)
        
        mse_scores = torch.cat(mse_scores, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # 简单评估：使用中位数作为阈值划分正常/异常
        threshold = mse_scores.median()
        preds = (mse_scores > threshold).long()
        accuracy = torch.eq(preds, targets).sum().item() / targets.size(0)
        return accuracy, mse_scores, targets, np.mean(times)

    def run(self):
        # 监控负 Loss，因为 EarlyStopping 默认监控值变大
        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.patience, verbose=True, delta=self.configs.delta)
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader)
        
        t_loss, v_loss = [], []
        for epoch in range(self.configs.num_epochs):
            tr_l, _ = self.train(epoch + 1)
            val_l, _ = self.val(epoch + 1)
            t_loss.append(tr_l); v_loss.append(val_l)
            early_stopping(-val_l, self.checkpoint_path)
            if early_stopping.early_stop: break

        acc, scores, targets, t_mean = self.test()
        # 计算混淆矩阵
        threshold = scores.median()
        preds = (scores > threshold).long()
        conf_mat = get_confusion_matrix(preds, targets, 2)
        
        self.save_results(t_loss, [0]*len(t_loss), v_loss, [0]*len(v_loss), scores, targets, acc, conf_mat, t_mean)
        if self.accelerator.is_main_process:
        #     plot_loss_cruve(np.array(t_loss), v_loss, self.checkpoint_path)
            pass
        self.logging(self.time_now, acc, t_mean)

# =============================================================================
# 预训练实验类
# =============================================================================
class PreTrainingExperiment(BaseExperiment):

    def __init__(
        self, configs, accelerator: Accelerator, setting: str, time_now: str
    ) -> None:
        super().__init__(configs=configs, accelerator=accelerator, setting=setting)
        self.print_start_message(time_now=time_now)
        self.time_now = time_now

    def run(self) -> None:
        pass


def run_amc_experiment(configs, accelerator: Accelerator, setting: str, time_now: str) -> None:
    if configs.mode == "supervised":
        task = configs.task_name
        if task == 'AMC': Exp = AMCExperiment
        elif task == 'WTC': Exp = WTCExperiment 
        elif task == 'SS': Exp = SSExperiment
        elif task == 'AD': Exp = ADExperiment
        else: raise ValueError(f"Unknown task: {task}")
    elif configs.mode == "unsupervised":
        Exp = PreTrainingExperiment

    exp = Exp(configs=configs, setting=setting, accelerator=accelerator, time_now=time_now)
    exp.run()