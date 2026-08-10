from datetime import datetime
import random

from argparse import ArgumentParser
from accelerate import Accelerator

import numpy as np
import torch

from utils.experiment import run_amc_experiment


parser = ArgumentParser(description="PyTorch Auto Modulation Classification")

parser.add_argument(
    "--model",
    type=str,
    default="AMCNet",
    choices=[
        "AMCNet",
        "CDAT",
        "CTNet",
        "DenseCNN",
        "DP_DRSN",
        "EMC2Net",
        "InceptionTime",
        "MCformer",
        "MCLDNN",
        "ModernTCN",
        "MTAMR",
        "PETCGDNN",
    ],
    help="The model to be trained for Auto Modulation Classification",
)
parser.add_argument(
    "--mode",
    type=str,
    default="supervised",
    choices=["supervised", "unsupervised"],
    help="supervised or unsupervised learning for model training",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="RML2016a",
    choices=["RML2016a", "RML2016b", "RML2018a", "HisarMod2019.1"],
    help="The dataset to be used for Auto Modulation Classification",
)
parser.add_argument(
    "--snr",
    type=int,
    default=0,
    help="The signal-to-noise ratio (SNR) for the dataset",
)
parser.add_argument(
    "--root_path",
    type=str,
    default="./dataset",
    help="The path to the root directory of the dataset",
)
parser.add_argument(
    "--file_path",
    type=str,
    default="./dataset/hello.csv",
    help="The path of the training and testing dataset for supervised learning.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoints",
    help="The directory to save checkpoints.",
)
parser.add_argument(
    "--split_ratio",
    type=float,
    default=0.6,
    help="The ratio to split the trainning and testing dataset.",
)
parser.add_argument(
    "--patch_len",
    type=int,
    default=None,
    help="The length of each patch. Default comes from the model Config.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=None,
    help="The stride size when forming patches. Default comes from the model Config.",
)
parser.add_argument(
    "--scale", type=bool, default=True, help="Whether to standard the training data."
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=128,
    help="The length of each sequence of IQ inputs data.",
)

# The model hyper-parameters (default=None -> use each model's xxxConfig / scripts defaults)
parser.add_argument(
    "--d_model",
    type=int,
    default=None,
    help="The dimension of model for Transformer block. Default comes from the model Config.",
)
parser.add_argument(
    "--d_ff",
    type=int,
    default=None,
    help="The dimension of feedforward network. Default comes from the model Config.",
)
parser.add_argument(
    "--n_heads",
    type=int,
    default=None,
    help="The number of heads. Default comes from the model Config.",
)
parser.add_argument(
    "--n_layers",
    type=int,
    default=None,
    help="The number of encoder layers. Default comes from the model Config.",
)
parser.add_argument(
    "--activation",
    type=str,
    default=None,
    help="The activation function. Default comes from the model Config.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=None,
    help="The dropout rate in deep learning models. Default comes from the model Config.",
)

# The optimizer, scheduler, and criterion hyper-parameters
parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    help="The optimizer to use for training.",
    choices=["adam", "sgd", "adamw", "radam"],
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="OneCycle",
    help="The learning rate scheduler to use for training.",
    choices=["StepLR", "ExponLR", "CosineAnnealingLR", "OneCycle"],
)
# The loss function to use during training
parser.add_argument(
    "--criterion",
    type=str,
    default="mse",
    help="The loss function to use during training.",
    choices=["mse", "mae", "huber", "cross_entropy"],
)

# training hyper-parameters
parser.add_argument(
    "--batch_size", type=int, default=32, help="The batch size of training."
)
parser.add_argument(
    "--shuffle",
    type=bool,
    default=True,
    help="Whether to shuffle the training dataset.",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="The learning rate of optimizer."
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="The number of workers for dataloader."
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="The number of epochs to train the model.",
)
parser.add_argument(
    "--warmup_epochs", type=int, default=1, help="The number of warmup epochs."
)
parser.add_argument(
    "--warmup",
    type=str,
    default="linear",
    help="The warmup strategy: linear or constant.",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum size used in stochastic gradient descent",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="L2 regularization strength suitable for Adam",
)
parser.add_argument(
    "--beta1",
    type=float,
    default=0.9,
    help="Decay rate of first - order moment estimate, degree of retention of historical gradients, default 0.9",
)
parser.add_argument(
    "--beta2",
    type=float,
    default=0.999,
    help="Decay rate of second - order moment estimate, conducive to improving stability, default 0.999",
)
parser.add_argument(
    "--eps", type=float, default=1e-8, help="Constant to prevent division by zero"
)
parser.add_argument(
    "--amsgrad", type=bool, default=False, help="Whether to use the AMSgrad variant"
)
parser.add_argument(
    "--step_size",
    type=int,
    default=10,
    help="Number of Epochs in StepLR that multiply the learning rate by gamma at regular intervals",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Learning rate decay multiplier for StepLR and ExponLR",
)
parser.add_argument(
    "--cycle_momentum",
    type=bool,
    default=True,
    help="Whether to use periodic momentum adjustment strategy in OneCycle",
)
parser.add_argument(
    "--base_momentum",
    type=float,
    default=0.85,
    help="Base momentum value set during learning rate adjustment",
)
parser.add_argument(
    "--max_momentum",
    type=float,
    default=0.95,
    help="Momentum value set when learning rate reaches maximum",
)
parser.add_argument(
    "--div_factor",
    type=float,
    default=25.0,
    help="Initial learning rate divided by this factor for OneCycleLR",
)
parser.add_argument(
    "--final_div_factor",
    type=float,
    default=1e4,
    help="Minimum learning rate divided by this factor for OneCycleLR",
)
parser.add_argument(
    "--anneal_strategy",
    type=str,
    default="cos",
    help="Learning rate decay strategy used: cos or linear",
)

# early stopping parameters
parser.add_argument(
    "--patience", type=int, default=5, help="The patience for early stopping."
)
parser.add_argument(
    "--delta", type=float, default=0.0, help="The delta for early stopping."
)

# The config of the peft in LoRA fine-tuning
parser.add_argument(
    "--lora_r", type=int, default=8, help="The r parameter for LoRA fine-tuning."
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
    help="The alpha parameter for LoRA fine-tuning.",
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.00,
    help="The dropout parameter for LoRA fine-tuning.",
)

# The Fixed Random Seed
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator()

    # Get the current time for logging
    time_now = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")

    # Create the experiment setting config
    dm = args.d_model if args.d_model is not None else "cfg"
    df = args.d_ff if args.d_ff is not None else "cfg"
    setting = f"{args.model}_{args.dataset}_{args.snr}_{args.mode}_sl{args.seq_len}_bs{args.batch_size}_lr{args.learning_rate}_dm{dm}_df{df}_pat{args.patience}_sd{args.seed}_{time_now}"

    # Create and run the experiment
    exp = run_amc_experiment(
        configs=args, setting=setting, accelerator=accelerator, time_now=time_now
    )
