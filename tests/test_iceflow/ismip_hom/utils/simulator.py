#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import sys
from typing import Optional

import pytest
import tensorflow as tf

from igm.igm_run import main

RANDOM_SEED = 0
tf.config.experimental.enable_op_determinism()


def run_igm(
    monkeypatch: pytest.MonkeyPatch,
    length: Optional[int] = None,
    mapping: str = "identity",
    optimizer: str = "adam",
) -> str:
    """
    Run IGM simulation.

    Args:
        monkeypatch: pytest fixture for modifying sys.argv
        length: Length scale in km
        mapping: Mapping type
        optimizer: Optimizer type

    Returns:
        Path to output directory
    """
    experiment_optimizer = "lbfgs" if optimizer == "cg_newton" else optimizer
    argv = [
        "igm_run.py",
        f"+experiment=params_{experiment_optimizer}",
        f"processes.iceflow.unified.mapping={mapping}",
        f"processes.iceflow.unified.optimizer={optimizer}",
    ]

    if optimizer == "cg_newton":
        argv.extend(
            [
                "processes.iceflow.numerics.precision=double",
                "processes.iceflow.unified.cg_newton.hvp_mode=banded",
                "processes.iceflow.unified.cg_newton.probe_mode=autodiff",
                "processes.iceflow.unified.cg_newton.hvp_verify=false",
                "processes.iceflow.unified.cg_newton.preconditioner=barotropic_multigrid",
                "processes.iceflow.unified.cg_newton.multigrid.smoother_weight=0.5",
                "processes.iceflow.unified.cg_newton.damping=1e-15",
                "processes.iceflow.unified.cg_newton.damping_adaptive=false",
                "processes.iceflow.unified.cg_newton.cg_tol=1e-8",
                "processes.iceflow.unified.cg_newton.cg_max_iter=300",
                "processes.iceflow.unified.nbit_init=40",
                (
                    "processes.iceflow.unified.halt.success="
                    "[{criterion:rel_initial,metric:grad_theta_norm,"
                    "rel_initial:{tol:1e-8,ord:id}}]"
                ),
            ]
        )

    if length is not None:
        argv.append(f"inputs.init_state.L={length * 1e3}")
        path_run_dir = os.path.join("outputs", f"{length}km", mapping, optimizer)
    else:
        path_run_dir = os.path.join("outputs", mapping, optimizer)

    argv.append(f"hydra.run.dir={path_run_dir}")

    if mapping == "identity":
        argv.append("processes.iceflow.unified.adam.lr_init=0.9")

    monkeypatch.setattr(sys, "argv", argv)
    tf.random.set_seed(RANDOM_SEED)
    main()

    return path_run_dir
