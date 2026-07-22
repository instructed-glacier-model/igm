#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt
from ..energy_operator import (
    Operator,
    ADOperator,
    BandedADOperator,
)


class InterfaceCGNewton(InterfaceOptimizer):

    @staticmethod
    def _build_operator(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Operator:
        """Build the energy operator (grad J and v -> H v) from hvp_mode.

        - 'autodiff'        -> ADOperator (reverse-over-reverse AD; default).
                               Exact, general (any Nz), but CG pays a full
                               double-backward pass on EVERY inner iteration.
        - 'banded'          -> BandedADOperator. Extracts the exact 9-point
                               Hessian stencil once per Newton step by graph
                               colouring, then each CG iteration is a cheap
                               banded apply. SSA uses 18 probes; MOLHO repeats
                               the 9 spatial colours for all 2*Nz coupled U/V
                               components. Requires an identity mapping.
        """
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        precision = cfg_numerics.precision
        hvp_mode = cfg_unified.cg_newton.hvp_mode

        if hvp_mode == "autodiff":
            return ADOperator(cost_fn, map, precision)

        if hvp_mode == "banded":
            return BandedADOperator(
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
            "line_search_compile": cg.get("line_search_compile", True),
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            # --- CG solver -------------------------------------------------
            "cg_max_iter": cg.cg_max_iter,
            "cg_tol": cg.cg_tol,
            "warm_start": cg.warm_start,
            "damping": cg.damping,
            # --- Levenberg-Marquardt damping adaptation --------------------
            "damping_adaptive": cg.get("damping_adaptive", False),
            "damping_min": cg.get("damping_min", 1e-12),
            "damping_max": cg.get("damping_max", 1e2),
            "damping_down": cg.get("damping_down", 0.25),
            "damping_up": cg.get("damping_up", 4.0),
            # --- operator (grad J / v -> H v) ------------------------------
            "operator": operator,
            # --- preconditioning -------------------------------------------
            "preconditioner": cg.preconditioner,
            "precond_update_freq": cg.precond_update_freq,
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
