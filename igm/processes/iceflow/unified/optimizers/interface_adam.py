#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict

from ..mappings import Mapping
from .optimizer import Optimizer
from .interface import InterfaceOptimizer, Status


class InterfaceAdam(InterfaceOptimizer):

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified

        return {
            "cost_fn": cost_fn,
            "map": map,
            "lr": cfg_unified.lr,
            "iter_max": cfg_unified.nbit,
            "print_cost": cfg_unified.print_cost,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:

        cfg_unified = cfg.processes.iceflow.unified

        # only apply lr schedule if network mapping is used
        if hasattr(optimizer.map, "network"):
            lr_decay = cfg_unified.lr_decay
            lr_decay_steps = cfg_unified.lr_decay_steps
        else:
            lr_decay = 1.0
            lr_decay_steps = 1000000

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
            lr = cfg_unified.lr_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
            lr = cfg_unified.lr_init
        elif status == Status.DEFAULT:
            iter_max = cfg_unified.nbit
            lr = cfg_unified.lr
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        optimizer.update_parameters(
            iter_max=iter_max, lr=lr, lr_decay=lr_decay, lr_decay_steps=lr_decay_steps
        )

        return True
