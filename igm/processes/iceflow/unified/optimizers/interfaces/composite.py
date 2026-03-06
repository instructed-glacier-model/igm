#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable, Dict, Optional, Tuple

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping


class InterfaceComposite(InterfaceOptimizer):

    # Keys that are composite-specific and should not be merged into cfg_unified
    _STAGE_ONLY_KEYS = {"active", "optimizer"}

    @staticmethod
    def _merge_cfg_stage(cfg: DictConfig, cfg_stage: DictConfig) -> DictConfig:
        stage_dict = OmegaConf.to_container(cfg_stage, resolve=True)
        overrides = {
            k: v for k, v in stage_dict.items()
            if k not in InterfaceComposite._STAGE_ONLY_KEYS
        }
        cfg_updated = cfg.copy()
        cfg_updated.processes.iceflow.unified = OmegaConf.merge(
            cfg_updated.processes.iceflow.unified, overrides
        )
        return cfg_updated

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_composite = cfg_unified.composite
        cfg_numerics = cfg.processes.iceflow.numerics

        stages = [
            InterfaceComposite._build_stage(cfg, cost_fn, map, cfg_stage, save_args)
            for cfg_stage in cfg_composite.stages
        ]

        return {
            "cost_fn": cost_fn,
            "map": map,
            "stages": stages,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
        }

    @staticmethod
    def _build_stage(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        cfg_stage: DictConfig,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optimizer]:
        """Build a (active_mode, optimizer) pair for one stage."""

        # Lazy imports to avoid circular dependency with parent __init__
        from igm.processes.iceflow.unified.optimizers import (
            Optimizers,
            InterfaceOptimizers,
        )

        opt_name = cfg_stage.optimizer
        cfg_merged = InterfaceComposite._merge_cfg_stage(cfg, cfg_stage)
        opt_args = InterfaceOptimizers[opt_name].get_optimizer_args(
            cfg_merged, cost_fn, map, save_args
        )

        active = cfg_stage.get("active", "all")
        return (active, Optimizers[opt_name](**opt_args))

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        from igm.processes.iceflow.unified.optimizers import InterfaceOptimizers

        if status == Status.IDLE:
            return False

        cfg_unified = cfg.processes.iceflow.unified
        cfg_composite = cfg_unified.composite

        for stage, cfg_stage in zip(optimizer.stages, cfg_composite.stages):
            opt_name = cfg_stage.optimizer
            opt = stage["optimizer"]
            cfg_merged = InterfaceComposite._merge_cfg_stage(cfg, cfg_stage)
            InterfaceOptimizers[opt_name].set_optimizer_params(
                cfg_merged, status, opt
            )

        optimizer.update_parameters()
        return True
