#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt
from ...operators import (
    Operator,
    ADOperator,
    BandedADOperator,
    MOLHOBandedADOperator,
    SSABandedADOperator,
)
from ...operators import supports_compact_molho, supports_compact_ssa


class InterfaceCGNewton(InterfaceOptimizer):

    @staticmethod
    def _build_operator(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Operator:
        """Build the Hessian operator selected by ``hvp_mode``.

        Autodiff is exact and general. Banded mode freezes a graph-coloured
        9-point stencil for cheap CG applications; nonperiodic SSA and Nz=2
        MOLHO use specialized compact storage.
        """
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        precision = cfg_numerics.precision
        hvp_mode = cfg_unified.cg_newton.hvp_mode

        if hvp_mode == "autodiff":
            return ADOperator(cost_fn, map, precision)

        if hvp_mode == "banded":
            basis_vertical = str(cfg_numerics.get("basis_vertical", "")).lower()
            if supports_compact_ssa(map):
                operator_cls = SSABandedADOperator
            elif supports_compact_molho(map, basis_vertical):
                operator_cls = MOLHOBandedADOperator
            else:
                operator_cls = BandedADOperator

            return operator_cls(
                cost_fn,
                map,
                precision,
                verify_stencil=bool(
                    getattr(cfg_unified.cg_newton, "hvp_verify", False)
                ),
                probe_mode=str(
                    getattr(cfg_unified.cg_newton, "probe_mode", "fd")
                ),
            )

        raise ValueError(
            f"❌ Unknown cg_newton.hvp_mode: <{hvp_mode}>. "
            "Use 'autodiff' or 'banded'."
        )

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        cg = cfg_unified.cg_newton

        halt = Halt(**InterfaceHalt.get_halt_args(cfg))
        operator = InterfaceCGNewton._build_operator(cfg, cost_fn, map)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "iter_max": cfg_unified.nbit,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "line_search_method": cfg_unified.line_search,
            "line_search_compile": cg.get("line_search_compile", False),
            "print_timing": cg.get("print_timing", False),
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "cg_max_iter": cg.cg_max_iter,
            "cg_tol": cg.cg_tol,
            "warm_start": cg.warm_start,
            "damping": cg.damping,
            "damping_adaptive": cg.get("damping_adaptive", False),
            "damping_min": cg.get("damping_min", 1e-12),
            "damping_max": cg.get("damping_max", 1e2),
            "damping_down": cg.get("damping_down", 0.25),
            "damping_up": cg.get("damping_up", 4.0),
            "operator": operator,
            "preconditioner": cg.preconditioner,
            "operator_update_freq": cg.get("operator_update_freq", 1),
            "precond_update_freq": cg.precond_update_freq,
            "preconditioner_options": dict(cg.get("multigrid", {})),
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        cfg_unified = cfg.processes.iceflow.unified

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
        elif status == Status.DEFAULT:
            iter_max = cfg_unified.nbit
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        damping = cfg_unified.cg_newton.damping
        optimizer.update_parameters(iter_max=iter_max, damping=damping)

        return True
