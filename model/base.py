"""Shared helpers for AMC model configs."""

from __future__ import annotations

import inspect
from typing import Any, Type, TypeVar

from transformers import PretrainedConfig

T = TypeVar("T", bound=PretrainedConfig)


def build_config_from_experiment(config_cls: Type[T], configs: Any) -> T:
    """Build a model ``PretrainedConfig`` from experiment / CLI args.

    Values present on ``configs`` and not ``None`` override the model-specific
    defaults defined on ``config_cls``. Unspecified fields keep Config defaults
    (aligned with ``scripts/`` hyperparameters).
    """
    sig = inspect.signature(config_cls.__init__)
    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if not hasattr(configs, name):
            continue
        val = getattr(configs, name)
        if val is None:
            continue
        # argparse historically typed seq_len as float
        if name == "seq_len":
            val = int(val)
        kwargs[name] = val

    return config_cls(**kwargs)
