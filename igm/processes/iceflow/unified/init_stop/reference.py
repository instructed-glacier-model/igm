#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The direct solve the initial training is measured against (``unified.init_stop``)."""

import time
from dataclasses import dataclass

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from igm.common import State
from igm.processes.iceflow.utils.velocities import get_velsurf
from ..halt import HaltStatus
from ..mappings import InterfaceMappings, Mappings
from ..optimizers import InterfaceOptimizers, Optimizers, Status
from ..utils import get_cost_fn


@dataclass
class ReferenceSolve:
    """Surface speed of the direct solve, and how the solve went."""

    speed: tf.Tensor
    iterations: int
    seconds: float
    converged: bool


def surface_speed(U: tf.Tensor, V: tf.Tensor, V_s: tf.Tensor) -> tf.Tensor:
    """Surface speed [Ny, Nx] of the velocity coefficients U, V [Nz, Ny, Nx]."""
    uvelsurf, vvelsurf = get_velsurf(U, V, V_s)
    return tf.sqrt(tf.square(uvelsurf) + tf.square(vvelsurf))


def get_reference_cfg(cfg: DictConfig) -> DictConfig:
    """The run's configuration with the ice flow solved directly (identity + cg_newton).

    The solver settings are those of ``unified.cg_newton``, which the network does not use,
    with the CG tolerance of ``init_stop.reference``.
    """
    cfg_reference = cfg.processes.iceflow.unified.init_stop.reference

    halt = {
        "freq": 1,
        "raise_on_failure": True,
        "success": [
            {
                "criterion": "abs_change",
                "metric": "u",
                "abs_change": {
                    "tol": cfg_reference.tol,
                    "reduction": "rmse",
                    "consecutive": 3,
                },
            }
        ],
        "failure": [
            {"criterion": "nan", "metric": "u"},
            {"criterion": "inf", "metric": "u"},
        ],
    }
    overrides = {
        "mapping": "identity",
        "optimizer": "cg_newton",
        "line_search": cfg_reference.line_search,
        "nbit_init": cfg_reference.nbit,
        "cg_newton": {"cg_tol": cfg_reference.cg_tol},
        "halt": halt,
    }

    cfg_reference_run = cfg.copy()
    cfg_reference_run.processes.iceflow.unified = OmegaConf.merge(
        cfg_reference_run.processes.iceflow.unified, overrides
    )
    return cfg_reference_run


def solve_reference(cfg: DictConfig, state: State, inputs: tf.Tensor) -> ReferenceSolve:
    """Solve the ice flow of the current state directly, from a zero velocity."""
    cfg_reference_run = get_reference_cfg(cfg)

    mapping_args = InterfaceMappings["identity"].get_mapping_args(
        cfg_reference_run, state
    )
    mapping = Mappings["identity"](**mapping_args)

    interface = InterfaceOptimizers["cg_newton"]
    optimizer_args = interface.get_optimizer_args(
        cfg=cfg_reference_run,
        cost_fn=get_cost_fn(cfg_reference_run, state),
        map=mapping,
    )
    optimizer = Optimizers["cg_newton"](**optimizer_args)
    interface.set_optimizer_params(cfg_reference_run, Status.INIT, optimizer)

    start = time.perf_counter()
    costs = optimizer.minimize(inputs)
    U, V = mapping.get_UV(inputs)
    speed = surface_speed(U[0], V[0], state.iceflow.discr_v.V_s)
    speed.numpy()
    seconds = time.perf_counter() - start

    halt_state = optimizer.halt_state
    if halt_state is None:
        converged = False
    else:
        converged = int(halt_state.status) == HaltStatus.SUCCESS.value

    return ReferenceSolve(
        speed=speed,
        iterations=int(costs.shape[0]),
        seconds=seconds,
        converged=converged,
    )
