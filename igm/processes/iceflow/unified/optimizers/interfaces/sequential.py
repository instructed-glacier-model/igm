#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable, Dict, Optional

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping


class InterfaceSequential(InterfaceOptimizer):

    @staticmethod
    def _merge_cfg_stage(cfg: DictConfig, cfg_stage: DictConfig) -> DictConfig:
        cfg_updated = cfg.copy()
        cfg_updated.processes.iceflow.unified = OmegaConf.merge(
            cfg_updated.processes.iceflow.unified, cfg_stage
        )
        return cfg_updated

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Lazy import to avoid circular imports
        from .. import Optimizers, InterfaceOptimizers

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        optimizers = []
        for cfg_stage in cfg_unified.sequential.stages:
            optimizer_name = cfg_stage.optimizer.lower()

            cfg_merged = InterfaceSequential._merge_cfg_stage(cfg, cfg_stage)
            optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
                cfg_merged, cost_fn, map, save_args
            )
            optimizer = Optimizers[optimizer_name](**optimizer_args)
            optimizers.append(optimizer)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "optimizers": optimizers,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        # Lazy import to avoid circular imports
        from . import InterfaceOptimizers

        if status == Status.IDLE:
            return False

        cfg_unified = cfg.processes.iceflow.unified

        for optimizer_stage, cfg_stage in zip(
            optimizer.optimizers, cfg_unified.sequential.stages
        ):
            optimizer_name = cfg_stage.optimizer.lower()

            cfg_merged = InterfaceSequential._merge_cfg_stage(cfg, cfg_stage)
            InterfaceOptimizers[optimizer_name].set_optimizer_params(
                cfg_merged, status, optimizer_stage
            )

        return True
