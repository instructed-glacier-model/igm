#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from omegaconf import DictConfig

from igm.processes.iceflow.mappings import Mapping

from .. import Optimizer


class Status(Enum):
    INIT = auto()
    WARM_UP = auto()
    DEFAULT = auto()
    IDLE = auto()


class InterfaceOptimizer(ABC):

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
        t: Optional[float] = None,
    ) -> bool:
        """Set the optimizer's iterations and settings for this solve; return whether to solve.

        ``t`` is the model time of the step (None at initialisation), at which scheduled
        parameters such as ``nbit`` are read (``igm.utils.schedule.value_at``); it is only
        read for a schedule.
        """
        raise NotImplementedError(
            "❌ The set_optimizer_params static method is not implemented."
        )
