from typing import Optional, Union

import torch
from torch import nn


class SwiGLU(nn.Module):
    """The Gated Linear Unit with the Swish Function"""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None) -> None:
        """
        :param input_dim: (int) the input dimension for the linear proj in ``SwiGLU`` function.
        :param hidden_dim: Optional(int) the hidden dimension for the linear proj in ``SwiGLU`` function.
        """
        super(SwiGLU, self).__init__()

        # Determine the dimension of the hidden layer
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        # Create two linear transformations
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)

        # Using built-in Swish functions
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation part of the SwiGLU activation function.

        :param x: (Tensor) the output tensor to be activated.

        :return: (Tensor) the results of the activation.
        """
        return self.fc1(x) * self.swish(self.fc2(x))


class Activation(nn.Module):
    """
    Get the activation function to use according to the specified name.
    """

    def __init__(
        self,
        activation: Union[nn.Module, str] = "gelu",
        inplace: Optional[bool] = False,
        approximate: Optional[str] = "none",
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """
        :param activation: Union(nn.Module, str) the name of the activation function. Default: ``relu``.
        :param inplace: Optional(str) can optionally do the operation in-place. Default: ``False``
        :param approximate: Optional(str) the gelu approximation algorithm to use: ``None`` | ``tanH``.
        :param input_dim: Optional(int) the input dimension for the linear proj in ``SwiGLU`` function.
        :param hidden_dim: Optional(int) the hidden dimension for the linear proj in ``SwiGLU`` function.
        """
        super(Activation, self).__init__()

        # Determine whether the input is a directly callable object
        if callable(activation):
            self.activation = activation()

        # Determine whether it is a string
        assert isinstance(activation, str)
        self.name = activation.lower()

        # Select the activation function to use based on its name
        if self.name == "relu":
            self.activation = nn.ReLU(inplace=inplace)
        elif self.name == "gelu":
            self.activation = nn.GELU(approximate=approximate)
        elif self.name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.name == "logsigmoid":
            self.activation = nn.LogSigmoid()
        elif self.name == "swish":
            self.activation = nn.SiLU(inplace=inplace)
        elif self.name == "swiglu":
            self.activation = SwiGLU(input_dim, hidden_dim)
        else:
            raise ValueError("Please enter the correct activation function name!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation part of the activation function.

        :param x: (Tensor) the output tensor to be activated.

        :return: (Tensor) the results of the activation.
        """
        return self.activation(x)


class Transpose(nn.Module):
    """Transpose the dimensions of the input tensor"""

    def __init__(self, *dims, contiguous=False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class ReVIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) Module.
    A normalization technique designed to handle non-stationary time series data.
    This module normalizes the input data during the forward pass and denormalizes it during the reverse pass.
    The code is taken from the official implementation: https://github.com/ts-kim/RevIN/blob/master/RevIN.py
    Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(ReVIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x: torch.FloatTensor, norm: bool = True) -> torch.FloatTensor:
        """
        The forward method for RevIN.

        :param x: (Tensor) input tensor of shape [batch_size, seq_len, n_vars]
        :param norm: (bool) True for normalization, False for denormalization.

        :return: (Tensor) normalized or denormalized tensor with the same shape as input
        """
        if norm:
            self._get_statistics(x)
            x = self._normalize(x)
        else:
            x = self._denormalize(x)
        return x

    def _init_params(self) -> None:
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.FloatTensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        The normalization process of RevIN.
        """
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        The denormalization process of RevIN.
        """
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class RMSNorm(torch.nn.Module):
    """
    The RMSNorm layer implementation from the paper `Root Mean Square Layer Normalization`.
    This layer normalizes the inputs based on the root mean square (RMS) of the input values.
    The code is taken from the `huggingface transformers library`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        :param hidden_size: (int) The size of the hidden layer.
        :param eps: (float) A small value to avoid division by zero, default is 1e-6.
        """
        super(RMSNorm, self).__init__()
        # Learnable weight parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Epsilon value for numerical stability
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the RMSNorm layer.

        :param x: (Tensor) Input tensor of shape (..., hidden_size).

        :return: (Tensor) Normalized tensor of the same shape as input.
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute the variance (mean of squares) along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        # Scale the normalized tensor with the learnable weight parameter
        return self.weight * x.to(input_dtype)


class Normalization(nn.Module):
    """
    Normalization layer used including ``BatchNorm``, ``LayerNorm`` and ``RMSNorm``.
    """

    def __init__(
        self,
        num_features: int,
        norm_cls: Union[str, nn.Module] = "batchnorm",
        eps: Optional[float] = 1e-6,
    ) -> None:
        super(Normalization, self).__init__()

        self.num_features = num_features

        if isinstance(norm_cls, nn.Module):
            self.normalization = norm_cls
            self.norm_name = norm_cls.__class__.__name__.lower()

        elif isinstance(norm_cls, str):
            self.norm_name = norm_cls.lower()
            if self.norm_name == "batchnorm":
                self.normalization = nn.Sequential(
                    Transpose(1, 2), nn.BatchNorm1d(num_features), Transpose(1, 2)
                )
            elif self.norm_name == "layernorm":
                self.normalization = nn.LayerNorm(num_features, eps=eps)
            elif self.norm_name == "rmsnorm":
                self.normalization = RMSNorm(hidden_size=num_features, eps=eps)

            else:
                raise ValueError

        else:
            raise ValueError(f"Unsupported norm type: {type(norm_cls)}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        标准化层的正向传播部分

        :param features:

        :return: (Tensor)
        """
        return self.normalization(features)
