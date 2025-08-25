#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict

from ..mappings import Mapping
from ..optimizers import Optimizer


class Status(Enum):
    INIT = auto()
    WARM_UP = auto()
    DEFAULT = auto()
    IDLE = auto()


class Interface(ABC):

    @staticmethod
    @abstractmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "❌ The get_optimizer_args static method is not implemented."
        )

    @staticmethod
    @abstractmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        raise NotImplementedError(
            "❌ The set_optimizer_params static method is not implemented."
        )
