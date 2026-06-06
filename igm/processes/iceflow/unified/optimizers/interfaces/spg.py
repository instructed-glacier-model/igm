#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from typing import Any, Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt

try:
    from ...mappings import MappingDataAssimilation, MappingCombinedDataAssimilation
except Exception:  # pragma: no cover - keeps the interface compatible across IGM branches.
    try:
        from ...mappings.data_assimilation import MappingDataAssimilation
    except Exception:  # pragma: no cover
        MappingDataAssimilation = None
    MappingCombinedDataAssimilation = None


def _is_data_assimilation_mapping(map: Mapping) -> bool:
    da_types = tuple(
        t for t in (MappingDataAssimilation, MappingCombinedDataAssimilation) if t is not None
    )
    return bool(da_types) and isinstance(map, da_types)


def _select(cfg: DictConfig, path: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def _spg_path(cfg: DictConfig, map: Mapping) -> str:
    da_path = "assimilations.field_inversion.optimization.spg"
    if _is_data_assimilation_mapping(map) and OmegaConf.select(cfg, da_path, default=None) is not None:
        return da_path
    return "processes.iceflow.unified.spg"


class InterfaceSPG(InterfaceOptimizer):
    """Configuration interface for OptimizerSpectralProjectedGradient."""

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        spg_path = _spg_path(cfg, map)

        if _is_data_assimilation_mapping(map):
            iter_max = int(
                _select(
                    cfg,
                    "assimilations.field_inversion.optimization.nbitmax",
                    cfg_unified.nbit,
                )
            )
        else:
            iter_max = int(cfg_unified.nbit)

        halt_args = InterfaceHalt.get_halt_args(cfg)
        halt = Halt(**halt_args)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "iter_max": iter_max,
            "alpha0": float(_select(cfg, f"{spg_path}.alpha0", 1.0)),
            "alpha_min": float(_select(cfg, f"{spg_path}.alpha_min", 1e-12)),
            "alpha_max": float(_select(cfg, f"{spg_path}.alpha_max", 1e12)),
            "armijo_c": float(_select(cfg, f"{spg_path}.armijo_c", 1e-4)),
            "backtrack_factor": float(_select(cfg, f"{spg_path}.backtrack_factor", 0.5)),
            "max_backtracks": int(_select(cfg, f"{spg_path}.max_backtracks", 20)),
            "nonmonotone_window": int(_select(cfg, f"{spg_path}.nonmonotone_window", 10)),
            "bb_variant": str(_select(cfg, f"{spg_path}.bb_variant", "alternating")),
            "step_tol": float(_select(cfg, f"{spg_path}.step_tol", 0.0)),
            "use_da_display": bool(_select(cfg, f"{spg_path}.use_da_display", True)),
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        cfg_unified = cfg.processes.iceflow.unified
        spg_path = _spg_path(cfg, optimizer.map)

        if status == Status.INIT:
            iter_max = int(cfg_unified.nbit_init)
            alpha0 = float(_select(cfg, f"{spg_path}.alpha0_init", _select(cfg, f"{spg_path}.alpha0", 1.0)))
        elif status == Status.WARM_UP:
            iter_max = int(cfg_unified.nbit_init)
            alpha0 = float(_select(cfg, f"{spg_path}.alpha0_init", _select(cfg, f"{spg_path}.alpha0", 1.0)))
        elif status == Status.DEFAULT:
            iter_max = int(cfg_unified.nbit)
            alpha0 = float(_select(cfg, f"{spg_path}.alpha0", 1.0))
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        optimizer.update_parameters(
            iter_max=iter_max,
            alpha0=alpha0,
            alpha_min=float(_select(cfg, f"{spg_path}.alpha_min", 1e-12)),
            alpha_max=float(_select(cfg, f"{spg_path}.alpha_max", 1e12)),
        )
        return iter_max > 0
