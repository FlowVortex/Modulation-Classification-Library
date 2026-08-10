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

from transformers import PretrainedConfig

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
    IOformerConfig,
    IOformerModel,
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
    NMformerConfig,
    NMformerModel,
    PETCGDNNConfig,
    PETCGDNNModel,
    RadioLLMConfig,
    RadioLLMModel,
    SpectrumFMConfig,
    SpectrumFMModel,
    build_config_from_experiment,
)

from utils.dataset import (
    RML2016aDataLoader,
    RML2016bDataLoader,
    RML2018aDataLoader,
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

        # The mode of the experiment: "supervised" or "pre-training"
        self.mode = configs.mode

        # The hyper-parameters for data loading
        self.batch_size = configs.batch_size
        self.shuffle = configs.shuffle

        # The checkpoint directory
        self.checkpoint_dir = configs.checkpoint
        self.checkpoint_path = path.join(self.checkpoint_dir, setting)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Get the number of classes in the dataset
        self.n_classes = None
        self.class_list = None

        # The file path of the training dataset for supervised learning
        # Or the testing dataset for unsupervised learning (zero-shot learning)
        self.file_path = configs.file_path

        # The root path of the data for model fine-tuning or large scale pre-training
        self.root_path = configs.root_path

        # The number of epochs for early stopping
        self.patience = configs.patience

    @abstractmethod
    def run(self, setting: str) -> None:
        """Run the experiment."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def model_dict(self) -> Dict[str, Tuple[type, type]]:
        """Return a dictionary of available (Config, Model) pairs."""
        return {
            "MCformer": (MCformerConfig, MCformerModel),
            "AMCNet": (AMCNetConfig, AMCNetModel),
            "CDAT": (CDATConfig, CDATModel),
            "CTNet": (CTNetConfig, CTNetModel),
            "DenseCNN": (DenseCNNConfig, DenseCNNModel),
            "DP_DRSN": (DP_DRSNConfig, DP_DRSNModel),
            "EMC2Net": (EMC2NetConfig, EMC2NetModel),
            "IOformer": (IOformerConfig, IOformerModel),
            "InceptionTime": (InceptionTimeConfig, InceptionTimeModel),
            "MCLDNN": (MCLDNNConfig, MCLDNNModel),
            "MTAMR": (MTAMRConfig, MTAMRModel),
            "NMformer": (NMformerConfig, NMformerModel),
            "PETCGDNN": (PETCGDNNConfig, PETCGDNNModel),
            "ModernTCN": (ModernTCNConfig, ModernTCNModel),
            "RadioLLM": (RadioLLMConfig, RadioLLMModel),
            "SpectrumFM": (SpectrumFMConfig, SpectrumFMModel),
        }

    def build_model_config(self, name: str) -> PretrainedConfig:
        """Build a model-specific ``PretrainedConfig`` from experiment configs.

        Model defaults come from each ``xxxConfig`` (aligned with ``scripts/``).
        Non-``None`` values from ``main.py`` / CLI args override those defaults.
        """
        assert name in self.model_dict, f"Model {name} is not supported."
        config_cls, _ = self.model_dict[name]
        return build_config_from_experiment(config_cls, self.configs)

    def build_model(self, name: str = "AMCNet") -> nn.Module:
        """Build the model for training via its ``xxxConfig``."""
        assert name in self.model_dict, f"Model {name} is not supported."

        self.accelerator.print(f"Building the model: {name}", end=" -> ")

        _, model_cls = self.model_dict[name]
        model_config = self.build_model_config(name)
        model = model_cls(model_config)

        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)
        return model

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any, Any]:
        """Load the dataset for training, validation, and testing."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_optimizer(self):
        """
        Load the optimizer for training.

        Creating an optimizer requires passing in the model's parameters.
        """
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        if self.optim is not None:
            return self.optim.load_optimizer(parameters=self.get_trainable_params())
        else:
            raise ValueError("Optimizer interface is not initialized.")

    def _load_scheduler(self, optimizer: Optimizer, loader_len: int):
        """
        Load the learning rate scheduler for training.
        """
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."

        if self.optim is not None:
            return self.optim.load_scheduler(optimizer, loader_len)
        else:
            raise ValueError("Optimizer interface is not initialized.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Obtain trainable parameters of the model in experiment."""
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Obtain the number of trainable parameters"""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_learning_rate(self, optimizer: Optimizer) -> float:
        """Get the current learning rate of the optimizer"""
        return optimizer.param_groups[0]["lr"]

    def _check_loss(self, loss: torch.FloatTensor) -> None:
        """Check the training and validation losses to avoid gradient explosion."""
        if not torch.isfinite(loss):
            self.accelerator.print(
                Fore.RED
                + "WARNING: non-finite loss, ending training!"
                + Style.RESET_ALL
            )
            sys.exit(1)

    def print_start_message(self, time_now: str) -> None:
        """Prints a message indicating the start of the experiment."""
        print_configs(
            accelerator=self.accelerator,
            time_now=time_now,
            config={  # The main configs parameters to be printed
                "seq_len": self.configs.seq_len,
                "epochs": self.configs.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.configs.learning_rate,
                "optimizer": self.configs.optimizer,
                "scheduler": self.configs.scheduler,
                "criterion": self.configs.criterion,
            },
            experiment_name="Auto Modulation Classification",
            model_name=self.model_name,
            dataset=self.configs.dataset,
            mode=self.mode,
            print_separator=True,
        )

    def save_results(
        self,
        train_loss: Union[np.ndarray, torch.Tensor],
        train_acc: Union[np.ndarray, torch.Tensor],
        val_loss: Union[np.ndarray, torch.Tensor],
        val_acc: Union[np.ndarray, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        accuracy: Union[float, torch.Tensor, np.ndarray],
        confusion_matrix: Union[torch.Tensor, np.ndarray],
        time_mean: Union[float, np.ndarray],
    ) -> None:
        """Save experimental results to a specified path."""
        results_path = self.checkpoint_path + "/results.pth"

        # Wait for all processes to finish before saving
        self.accelerator.wait_for_everyone()

        # Only main process saves the results
        if self.accelerator.is_main_process:
            # Save the model inputs, predictions, targets, MSE, and MAE
            self.accelerator.save(
                obj={
                    "train_loss": torch.tensor(train_loss),
                    "train_acc": torch.tensor(train_acc),
                    "val_loss": torch.tensor(val_loss),
                    "val_acc": torch.tensor(val_acc),
                    "predictions": predictions,
                    "targets": targets,
                    "accuracy": torch.tensor(accuracy),
                    "confusion_matrix": confusion_matrix,
                    "time_mean": torch.tensor(time_mean),
                },
                f=results_path,
                safe_serialization=True,
            )
        self.accelerator.print("Test results saved to " + results_path)

    @property
    def logging_headers(self) -> List[str]:
        """The Log header for the results of the csv."""
        return [
            "Timestamp",
            "Dataset",
            "Model",
            "SNR",
            "Accuracy",
            "Time",
            "split_ratio",
            "seq_len",
            "n_classes",
            "criterion",
            "optimizer",
            "learning_rate",
            "batch_size",
            "setting",
            "seed",
            "path",
        ]

    def logging(
        self, time_now: str, accuracy: float, time_mean: Union[float, np.ndarray]
    ) -> None:
        # Log the experiment results
        # Wait for all processes to finish before saving
        self.accelerator.wait_for_everyone()

        # Only save main process saves the model
        if self.accelerator.is_main_process:

            messages = {
                "Timestamp": time_now,
                "Dataset": self.configs.dataset,
                "Model": self.model_name,
                "SNR": self.configs.snr,
                "Accuracy": accuracy,
                "Time": time_mean,
                "split_ratio": self.configs.split_ratio,
                "seq_len": self.configs.seq_len,
                "n_classes": self.configs.n_classes,
                "criterion": self.configs.criterion,
                "optimizer": self.configs.optimizer,
                "learning_rate": self.configs.learning_rate,
                "batch_size": self.configs.batch_size,
                "setting": self.setting,
                "seed": self.configs.seed,
                "path": self.checkpoint_path,
            }

            # logging the results to csv file
            logging_results(
                accelerator=self.accelerator,
                logging_path="./results.csv",
                headers=self.logging_headers,
                messages=messages,
            )

            # Print the state
            self.accelerator.print(
                Fore.GREEN + "The results logging is done." + Fore.RESET
            )


class SupervisedExperiment(BaseExperiment):

    def __init__(
        self, configs, accelerator: Accelerator, setting: str, time_now: str
    ) -> None:
        super().__init__(configs=configs, accelerator=accelerator, setting=setting)
        self.print_start_message(time_now=time_now)
        self.time_now = time_now

        # Load the data for training and testing
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        # Build the deep learning model
        self.model_name = configs.model
        self.model = self.build_model(name=self.model_name)

        # Load the optimizer and scheduler for model training
        self.optim = OptimInterface(configs=configs, accelerator=accelerator)
        self.optimizer = self._load_optimizer()
        self.scheduler = self._load_scheduler(
            optimizer=self.optimizer,
            loader_len=len(self.train_loader),
        )

        # Create the loss function for classification
        self.criterion = get_loss_fn(configs.criterion)

        # The epoch number for model training
        self.num_epochs = configs.num_epochs

    @property
    def dataset_dict(
        self,
    ) -> Dict[str, Union[RML2016aDataLoader, RML2016aDataLoader, RML2018aDataLoader]]:
        """Return the dictionary of dataset loaders."""
        return {
            "RML2016a": RML2016aDataLoader,
            "RML2016b": RML2016bDataLoader,
            "RML2018a": RML2018aDataLoader,
            # "HisarMod2019.1": None,
        }

    def load_data(
        self,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Load the data for training, validation and testing in auto modulation classification."""

        assert (
            self.dataset in self.dataset_dict.keys()
        ), f"{self.dataset} is not in {self.dataset_list}!"

        data_interface = self.dataset_dict[self.dataset](configs=self.configs)
        self.class_list = data_interface.class_list
        self.n_classes = len(self.class_list)
        self.configs.n_classes = self.n_classes

        train_loader, val_loader, test_loader = data_interface.load()

        return train_loader, val_loader, test_loader

    def train(
        self, epoch: int
    ) -> Tuple[
        Union[float, torch.Tensor, np.ndarray], Union[float, torch.Tensor, np.ndarray]
    ]:
        """
        A unified approach for training self-supervised models one epoch.

        We will use a loop to repeatedly call this method to alternate between training and validating the model.
        """
        # Set the model to train mode
        self.model.train()
        num_samples = 0

        # Loop over the batches
        train_loss = torch.zeros(1, device=self.accelerator.device)
        train_accuracy = torch.zeros(1, device=self.accelerator.device)

        data_loader = tqdm(self.train_loader, file=sys.stdout)

        for step, (batch_x, batch_y) in enumerate(data_loader, 1):
            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            num_samples += batch_y.size(0)

            loss = self.criterion(outputs, batch_y)
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()

            _, predicted = torch.max(outputs, dim=1)
            train_accuracy += torch.eq(predicted, batch_y).sum()

            train_loss += loss.item()

            data_loader.desc = f"[Train Epoch {epoch}] Loss: {round(train_loss.item() / num_samples, 4)}, Accuracy: {round(train_accuracy.item() / num_samples, 4)}"

        self._check_loss(loss=train_loss)

        return train_loss.item() / num_samples, train_accuracy.item() / num_samples

    def val(
        self, epoch: int
    ) -> Tuple[
        Union[float, torch.Tensor, np.ndarray], Union[float, torch.Tensor, np.ndarray]
    ]:
        """
        Validation methods for self-supervised learning-based spectrum prediction models in each epoch.

        We use this method to determine whether an early stop operation needs to be performed.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize validation loss
        val_loss = torch.zeros(1, device=self.accelerator.device)
        val_accuracy = torch.zeros(1, device=self.accelerator.device)

        num_samples = 0

        # Iterate over validation data
        data_loader = tqdm(self.val_loader, file=sys.stdout)

        with torch.no_grad():
            # Iterate over batches
            for step, (batch_x, batch_y) in enumerate(data_loader, 1):
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                num_samples += batch_y.size(0)

                loss = self.criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                val_accuracy += torch.eq(predicted, batch_y).sum()

                # accumulate the metrics for time series forecasting
                data_loader.desc = f"[Test  Epoch {epoch}] Loss: {round(val_loss.item() / num_samples, 4)}, Accuracy: {round(val_accuracy.item() / num_samples, 4)}"

        return val_loss.item() / num_samples, val_accuracy.item() / num_samples

    def test(self) -> Tuple[float, torch.Tensor, torch.Tensor, Union[np.ndarray]]:
        """
        A unified interface for auto modulation classification testing.

        To ensure the reliability of the experiment,
        all base classes used for verifying results uniformly call this method to obtain the results.

        The created `data_loader` object in PyTorch needs to be passed in.
        """
        # loading the model parameters from checkpoint
        self.accelerator.print(
            "🚀 "
            + Fore.BLUE
            + "Loading the best model for testing..."
            + Style.RESET_ALL
        )
        self.accelerator.load_state(self.checkpoint_path)

        # Create the lists to store predictions and targets
        predictions = []
        targets = []
        time_list = []

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader):

                time_start = time.time()
                outputs = self.model(batch_x)
                time_end = time.time()

                time_list.append(time_end - time_start)

                predictions.append(outputs)
                targets.append(batch_y)

        predictions = torch.concatenate(predictions, axis=0)
        targets = torch.concatenate(targets, axis=0)

        # Calculate the evaluation metrics
        num_samples = predictions.shape[0]

        _, predicted = torch.max(predictions, dim=1)
        accuracy = torch.eq(predicted, targets).sum() / num_samples

        return accuracy.item(), predictions, targets, np.mean(time_list)

    def run(self) -> None:
        # Create the EarlyStopping object
        early_stopping = EarlyStopping(
            accelerator=self.accelerator,
            patience=self.patience,
            verbose=True,
            delta=self.configs.delta,
        )

        # Prepare the model, optimizer, and data loaders with accelerator
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        )

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        for epoch in range(self.num_epochs):
            # Training the model
            train_loss_epoch, train_acc_epoch = self.train(epoch=epoch + 1)
            train_loss.append(train_loss_epoch)
            train_acc.append(train_acc_epoch)

            # Validation
            val_loss_epoch, val_acc_epoch = self.val(epoch=epoch + 1)
            val_loss.append(val_loss_epoch)
            val_acc.append(val_acc_epoch)

            # Check the early stopping condition
            early_stopping(
                accuracy=val_acc_epoch,
                checkpoint_path=self.checkpoint_path,
            )

            if early_stopping.early_stop:
                self.accelerator.print(
                    "🔥"
                    + Fore.RED
                    + "Early stopping triggered. Stopping training."
                    + Style.RESET_ALL
                )
                break

        # Start testing after training
        accuracy, predictions, targets, time_mean = self.test()

        # Calculate the confusion matrix
        confusion_matrix = get_confusion_matrix(
            predictions=torch.max(predictions, dim=1)[1],
            targets=targets,
            n_classes=self.n_classes,
        )

        # save the all results in experiment
        self.save_results(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            predictions=predictions,
            targets=targets,
            confusion_matrix=confusion_matrix,
            time_mean=time_mean,
            accuracy=accuracy,
        )

        # Plot the experiment results
        self.accelerator.print(
            Fore.RED + "Plotting the experiment results" + Style.RESET_ALL, end=" -> "
        )

        if self.accelerator.is_main_process:
            # 在主进程中进行可视化
            plot_loss_cruve(
                train_loss=np.asarray(train_loss),
                val_loss=val_loss,
                save_path=self.checkpoint_path,
            )
            plot_accuracy_curve(
                train_acc=np.asarray(train_acc),
                val_acc=val_acc,
                save_path=self.checkpoint_path,
            )
            plot_confusion_matrix(
                confusion_matrix=np.asarray(confusion_matrix),
                save_path=self.checkpoint_path,
            )
        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)

        # logging the results to csv file
        self.logging(time_now=self.time_now, accuracy=accuracy, time_mean=time_mean)


class PreTrainingExperiment(BaseExperiment):

    def __init__(
        self, configs, accelerator: Accelerator, setting: str, time_now: str
    ) -> None:
        super().__init__(configs=configs, accelerator=accelerator, setting=setting)
        self.print_start_message(time_now=time_now)
        self.time_now = time_now

    def run(self) -> None:
        pass


def run_amc_experiment(
    configs, accelerator: Accelerator, setting: str, time_now: str
) -> None:
    """The General experiment interface for the Auto Modulation Classification task."""

    # Determine the different patterns
    mode = configs.mode

    if mode == "supervised":
        Exp = SupervisedExperiment
    elif mode == "unsupervised":
        Exp = PreTrainingExperiment

    # Create and run the experiment
    exp = Exp(
        configs=configs, setting=setting, accelerator=accelerator, time_now=time_now
    )

    # Start the experiment
    exp.run()
