#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping, MappingDataAssimilation, MappingCombinedDataAssimilation
from ...halt import Halt, InterfaceHalt


class InterfaceAdam(InterfaceOptimizer):

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        if isinstance(map, MappingDataAssimilation) or isinstance(
            map, MappingCombinedDataAssimilation
        ):
            lr = cfg.processes.SR_DA.optimization.learning_rate

        else:
            lr = cfg_unified.adam.lr

        halt_args = InterfaceHalt.get_halt_args(cfg)
        halt = Halt(**halt_args)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "lr": lr,
            "iter_max": cfg_unified.nbit,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "clip_norm": cfg_unified.adam.optimizer_clipnorm,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "batch_size": cfg_unified.data_preparation.patches_per_batch,
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
            lr_decay = cfg_unified.adam.lr_decay
            lr_decay_steps = cfg_unified.adam.lr_decay_steps
        else:
            lr_decay = 1.0
            lr_decay_steps = 1000000

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
            lr = cfg_unified.adam.lr_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
            lr = cfg_unified.adam.lr_init
        elif status == Status.DEFAULT:
            iter_max = cfg_unified.nbit
            lr = cfg_unified.adam.lr
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        optimizer.update_parameters(
            iter_max=iter_max, lr=lr, lr_decay=lr_decay, lr_decay_steps=lr_decay_steps
        )

        return True
