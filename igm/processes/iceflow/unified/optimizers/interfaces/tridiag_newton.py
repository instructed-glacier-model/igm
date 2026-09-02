#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
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
    Tridiag1DADOperator,
    Tridiag1DAnalyticOperator,
    supports_tridiag1d,
)
from igm.processes.iceflow.energy.utils import get_energy_components


class InterfaceTridiagNewton(InterfaceOptimizer):

    @staticmethod
    def _build_operator(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Operator:
        """Build the direct block-tridiagonal operator for a y-invariant grid.

        Requires ``Ny=2`` with a periodic north-south boundary and a
        non-periodic x axis (SSA, ``basis_vertical: ssa``) — see
        ``tridiag1d.supports_tridiag1d``.
        """
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        precision = cfg_numerics.precision
        tn = cfg_unified.tridiag_newton

        if not supports_tridiag1d(map):
            raise ValueError(
                "❌ optimizer: tridiag_newton requires an identity SSA mapping "
                "with shape (B, 1, 2, Nx), periodic in y and non-periodic in "
                "x (bcs including periodic_ns but not periodic_we, "
                "numerics.basis_vertical: ssa)."
            )

        assembly = str(getattr(tn, "assembly", "probe")).strip().lower()
        if assembly == "analytic":
            return Tridiag1DAnalyticOperator(
                cost_fn,
                map,
                cfg,
                get_energy_components(cfg),
                precision,
            )
        if assembly != "probe":
            raise ValueError(
                "tridiag_newton.assembly must be 'probe' or 'analytic', "
                f"got {assembly!r}."
            )

        return Tridiag1DADOperator(
            cost_fn,
            map,
            precision,
            probe_mode=str(getattr(tn, "probe_mode", "autodiff")),
        )

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        tn = cfg_unified.tridiag_newton

        halt = Halt(**InterfaceHalt.get_halt_args(cfg))
        operator = InterfaceTridiagNewton._build_operator(cfg, cost_fn, map)

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
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "damping": tn.damping,
            "damping_adaptive": tn.get("damping_adaptive", False),
            "damping_min": tn.get("damping_min", 1e-12),
            "damping_max": tn.get("damping_max", 1e2),
            "damping_down": tn.get("damping_down", 0.25),
            "damping_up": tn.get("damping_up", 4.0),
            "operator": operator,
            "scalar_flowline": tn.get("scalar_flowline", False),
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

        damping = cfg_unified.tridiag_newton.damping
        optimizer.update_parameters(iter_max=iter_max, damping=damping)

        return True
