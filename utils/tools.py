from typing import List, Optional, Tuple, Dict, Union, Any

import os
from os import path
import csv
from pathlib import Path

from colorama import Fore, Style
from accelerate import Accelerator

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from seaborn import heatmap

import torch
from torch import nn

from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class OptimInterface(object):
    """
    The General Interface for Loading Optimizers,
    including Learning Rate Warmup and Dynamic Learning Rate Adjustment
    """

    def __init__(self, configs, accelerator: Accelerator) -> None:
        self.accelerator = accelerator
        self.configs = configs
        # Get the optimizer used
        self.optimizer = configs.optimizer

        # Methods for obtaining predictions and dynamic learning rate adjustment
        self.warmup, self.scheduler = configs.warmup, configs.scheduler

        # Get the number of warm-up rounds and the total number of training rounds
        self.num_epochs, self.warmup_epochs = configs.num_epochs, configs.warmup_epochs
        self.pct_start = self.warmup_epochs / self.num_epochs

        # Get optimizer configuration parameters
        self.learning_rate = configs.learning_rate
        self.momentum = configs.momentum
        self.weight_decay = configs.weight_decay
        self.beta1, self.beta2 = configs.beta1, configs.beta2
        self.eps = configs.eps

        # whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)
        self.amsgrad = configs.amsgrad

        # Parameters for dynamic learning rate adjustment
        self.step_size = configs.step_size
        self.gamma = configs.gamma
        self.cycle_momentum = configs.cycle_momentum
        self.base_momentum = configs.base_momentum
        self.max_momentum = configs.max_momentum
        self.anneal_strategy = configs.anneal_strategy

    def load_optimizer(self, parameters: Optional[torch.Tensor | List]) -> Optimizer:
        """How to get the optimizer"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the optimizer: {self.optimizer}"
            + Style.RESET_ALL,
            end=" -> ",
        )

        name = self.optimizer.lower()

        if name == "sgd":
            # Using stochastic gradient descent
            return self.load_SGD(parameters)

        elif name == "adam":
            # Using Adam optimizer
            return self.load_Adam(parameters)

        elif name == "adamw":
            # Using the AdamW optimizer
            return self.load_AdamW(parameters)

        elif name == "radam":
            # Using the RAdam optimizer
            return self.load_RAdam(parameters)

        else:
            raise ValueError("configs.optimizer inputs error!")

    def load_scheduler(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """Methods for obtaining dynamic learning rate adjustments"""
        self.accelerator.print(
            Fore.RED
            + f"Now is loading the scheduler: {self.scheduler}"
            + Style.RESET_ALL,
            end=" -> ",
        )
        # If OneCycle is used, it comes with a learning rate warm-up process
        if self.scheduler == "OneCycle":
            return self.load_OneCycleLR(optimizer, loader_len)

        # Reloading dynamic learning rate adjustment method
        if self.scheduler == "StepLR":
            dynamic_scheduler = self.load_StepLR(optimizer)
        elif self.scheduler == "ExponLR":
            dynamic_scheduler = self.load_ExponentialLR(optimizer)
        elif self.scheduler == "CosineAnnealingLR":
            dynamic_scheduler = self.load_CosineAnnealingLR(optimizer)
        else:
            raise ValueError("configs.scheduler inputs error!")

        if self.warmup == "None" or self.warmup == "False" or self.warmup_epochs == 0:
            return dynamic_scheduler
        #  load the learning rate warm-up method
        warmup_scheduler = self.load_warmup(optimizer)

        # Combining learning rate warmup and dynamic learning rate adjustment
        return lr_scheduler.SequentialLR(
            optimizer,
            [warmup_scheduler, dynamic_scheduler],
            milestones=[self.warmup_epochs, self.num_epochs],
        )

    def load_warmup(self, optimizer: Optimizer) -> LRScheduler:
        """Get the adjustment method of learning rate warm-up"""
        if self.warmup == "LinearLR":
            # Use linear learning rate growth
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0001,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.load_successfully()
            return scheduler
        else:
            raise ValueError("configs.warmup fill in error")

    def load_SGD(self, parameters: torch.Tensor) -> Union[Optimizer, torch.optim.SGD]:
        """Methods for obtaining a stochastic gradient descent optimizer"""
        optimizer = optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)
        self.load_successfully()
        return optimizer

    def load_Adam(self, parameters: torch.Tensor) -> Union[Optimizer, torch.optim.Adam]:
        """The Interface to Get the Adam optimizer"""
        optimizer = optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )
        self.load_successfully()
        return optimizer

    def load_AdamW(
        self, parameters: torch.Tensor
    ) -> Union[Optimizer, torch.optim.AdamW]:
        """The Interface to Get the AdamW optimizer"""
        optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )
        self.load_successfully()
        return optimizer

    def load_RAdam(
        self, parameters: torch.Tensor
    ) -> Union[Optimizer, torch.optim.RAdam]:
        """The Interface to Get the RAdam optimizer"""
        optimizer = optim.RAdam(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
        self.load_successfully()
        return optimizer

    def load_ExponentialLR(self, optimizer: Optimizer) -> LRScheduler:
        """Get the learning rate exponential decay factor"""
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        self.load_successfully()
        return scheduler

    def load_StepLR(self, optimizer: Optimizer) -> LRScheduler:
        """A method for obtaining dynamic learning rate attenuation for each certain number of Epochs in StepLR"""
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        self.load_successfully()
        return scheduler

    def load_OneCycleLR(
        self, optimizer: Optimizer, loader_len: int = None
    ) -> LRScheduler:
        """Obtaining a periodic cyclic dynamic learning rate adjustment method"""
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.configs.max_momentum,
            div_factor=self.configs.div_factor,
            final_div_factor=self.configs.final_div_factor,
            steps_per_epoch=loader_len,
            epochs=self.num_epochs,
        )
        self.load_successfully()
        return scheduler
    
    def load_CosineAnnealingLR(self, optimizer: Optimizer) -> LRScheduler:
        t_max = self.num_epochs - self.warmup_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
        self.load_successfully()
        return scheduler
    

    def load_successfully(self) -> None:
        """note that the optimizer / scheduler has been loaded successfully"""
        self.accelerator.print(Fore.GREEN + "successfully loaded!" + Style.RESET_ALL)


def get_loss_fn(name: str) -> nn.Module:
    """Get the loss function based on the name"""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")


class EarlyStopping(object):
    """
    The EarlyStopping class is used to stop training when the validation loss doesn't improve after a given number of epochs.
    This coda is modified from: Time-Series-Library.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        patience: int = 7,
        verbose: Optional[bool] = False,
        delta: Optional[float] = 0,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

        self.accelerator = accelerator

    def __call__(
        self,
        accuracy: Union[float, torch.Tensor, np.ndarray],
        checkpoint_path: str,
    ) -> None:
        score = accuracy
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(accuracy, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(accuracy, checkpoint_path)
            self.counter = 0

    def save_checkpoint(
        self, accuracy: Union[float, torch.Tensor, np.ndarray], checkpoint_path: str
    ) -> None:
        if self.verbose:
            self.accelerator.print(
                f"Validation loss decreased, accuracy: ({self.val_loss_min:.6f} --> {accuracy:.6f}).  Saving model ..."
            )

        # Wait for all processes to finish before saving
        self.accelerator.wait_for_everyone()

        # Only the main process saves the model
        if self.accelerator.is_main_process:
            self.accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        self.val_loss_min = accuracy


def logging_results(
    accelerator: Accelerator,
    logging_path: str,
    headers: List[str],
    messages: Dict[str, Union[str, float]],
) -> None:
    """
    Log training results to a CSV file, supporting multi-GPU training scenarios

    Args:
        accelerator: Accelerator object from the Accelerate library, used for multi-GPU synchronization and main process identification
        logging_path: Save path (including filename) for the CSV file
        headers: List of headers for the CSV file
        messages: Dictionary of results to be logged, where keys must correspond to the headers
    """
    # Perform file writing operations only in the main process to avoid multi-process conflicts
    if accelerator.is_main_process:
        # Check if the keys in messages are a subset of the headers
        message_keys = set(messages.keys())
        header_set = set(headers)
        if not message_keys.issubset(header_set):
            missing_keys = message_keys - header_set
            raise ValueError(
                f"Messages contain keys not present in headers: {missing_keys}"
            )

        # Ensure the parent directory exists
        os.makedirs(Path(logging_path).parent, exist_ok=True)

        # Determine if the CSV file already exists
        file_exists = Path(logging_path).exists()

        # Write to the CSV file
        with open(logging_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)

            # Write the header row if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write data in the order of headers (ensure consistent column order)
            # Filter and organize data according to the header sequence
            row_data = {header: messages.get(header, "") for header in headers}
            writer.writerow(row_data)

        # Synchronize all processes to ensure file writing is complete before other processes proceed
        accelerator.wait_for_everyone()


def print_configs(
    accelerator: Accelerator,
    time_now: str,
    config: Dict[str, Any],
    experiment_name: str = "Auto Modulation Classification",
    model_name: str = "SpectrumTime",
    dataset: str = "spec4",
    mode: str = "supervised",
    print_separator: bool = True,
) -> None:
    """
    Prints key configuration parameters for deep learning experiments, optimized for distributed training scenarios.
    This print configs function from wwhenxuan for the `Spectrum Prediction Library`.

    Args:
    -----
        accelerator: Accelerator instance from Hugging Face Accelerate library, used for unified printing in distributed training.
        time_now: Formatted string of the experiment start time (e.g., "2026-01-13 14:25:30").
        config: Dictionary containing all critical experiment parameters (e.g., lr, batch_size, epochs, etc.).
        experiment_name: Name of the experiment for title display (default: "Spectrum Prediction").
        model_name: Name of the model being used (default: "SpectrumTime").
        dataset: Name of the dataset for the experiment (default: "spec4").
        mode: Training mode (e.g., "supervised", "unsupervised", default: "supervised").
        print_separator: Whether to print separator lines for better readability (default: True).
    """
    # Build the content to be printed
    output_lines = []

    # Title and time information
    if print_separator:
        separator = "=" * 60
        output_lines.append(separator)
    output_lines.append(f"📋 {experiment_name} - Experiment Parameters")
    output_lines.append(f"🤖 {model_name} model ( {mode} mode ) on {dataset} dataset.")
    output_lines.append(f"⏰ Experiment Start Time: {time_now}")
    if print_separator:
        output_lines.append("-" * 60)

    # Calculate the maximum length of parameter names for aligned display
    max_key_len = max(len(str(key)) for key in config.keys()) if config else 0
    # Print all parameters with aligned formatting
    for key, value in config.items():
        # Format: parameter name left-justified, value left-aligned for readability
        line = f"🔧 {key.ljust(max_key_len)} : {value}"
        output_lines.append(line)

    if print_separator:
        output_lines.append(separator)

    # Concatenate all lines into a single text block
    output_text = "\n".join(output_lines)

    # Print to console via accelerator (supports distributed training)
    accelerator.print(output_text)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pass


def get_confusion_matrix(
    predictions: torch.Tensor, targets: torch.Tensor, n_classes: int
) -> None:
    """计算混淆矩阵"""
    # 展平并转为numpy数组
    predictions_np = predictions.flatten().cpu().numpy()
    targets_np = targets.flatten().cpu().numpy()

    # 计算混淆矩阵（labels确保类别顺序）
    labels = np.arange(n_classes) if n_classes else None
    cm_np = confusion_matrix(y_pred=predictions_np, y_true=targets_np, labels=labels)

    # 转回PyTorch张量
    cm_tensor = torch.from_numpy(cm_np)

    return cm_tensor


def plot_loss_cruve(
    train_loss: Union[torch.Tensor, np.ndarray],
    val_loss: Union[torch.Tensor, np.ndarray],
    save_path: str,
) -> None:
    """Plot the training and validation loss curves in Auto Modulation Classification."""

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Set grid style open
    ax.grid("--", color="gray", alpha=0.4)

    # Plot the loss curves
    ax.plot(train_loss, label="Training Loss", color="royalblue")
    ax.plot(val_loss, label="Validation Loss", color="darkorange")

    # Set legend and axis labels
    ax.legend(loc="upper right", fontsize=10.5)
    ax.set_xlabel("Running Epochs", fontsize=12)
    ax.set_ylabel("Classification Loss", fontsize=12)

    # Save the plot loss figure
    # fig.savefig(path.join(save_path, "loss_curve.png"), bbox_inches="tight", dpi=600)
    fig.savefig(path.join(save_path, "loss_curve.pdf"), bbox_inches="tight", dpi=800)


def plot_accuracy_curve(
    train_acc: Union[torch.Tensor, np.ndarray],
    val_acc: Union[torch.Tensor, np.ndarray],
    save_path: str,
) -> None:
    """Plot the training and validation accuracy curves in Auto Modulation Classification."""

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Set grid style open
    ax.grid("--", color="gray", alpha=0.4)

    # Plot the loss curves
    ax.plot(train_acc, label="Training Accuracy", color="royalblue")
    ax.plot(val_acc, label="Validation Accuracy", color="darkorange")

    # Set legend and axis labels
    ax.legend(loc="upper left", fontsize=10.5)
    ax.set_xlabel("Running Epochs", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    # Save the plot loss figure
    # fig.savefig(path.join(save_path, "accuracy_curve.png"), bbox_inches="tight", dpi=600)
    fig.savefig(
        path.join(save_path, "accuracy_curve.pdf"), bbox_inches="tight", dpi=800
    )


def plot_tsne():
    pass


def plot_confusion_matrix(
    confusion_matrix: Union[np.ndarray, torch.Tensor], save_path: str
) -> None:
    """Plot confusion matrix."""

    # Check if the confusion matrix is a torch Tensor
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.numpy()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plotting the heatmap according to the confusion matrix
    confusion = heatmap(
        confusion_matrix,
        square=True,
        annot=True,
        # fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        vmin=0,
        linecolor="k",
        linewidths=0.4,
        annot_kws={"size": 10},
    )

    # Set the colorbar
    cbar = confusion.figure.colorbar(
        confusion.collections[0], ax=confusion, location="right", shrink=0.8
    )
    # Set the axis labels
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    # Save the plotting results
    fig.savefig(
        path.join(save_path, "confusion_matrix.pdf"), bbox_inches="tight", dpi=800
    )
