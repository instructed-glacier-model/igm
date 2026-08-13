#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import sys
from igm.igm_run import main
from typing import Optional


def run_igm(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    length: Optional[int] = None,
    mapping: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> str:
    """
    Run IGM simulation for any method.

    Args:
        monkeypatch: pytest fixture for modifying sys.argv
        method: "unified", "emulated", or "solved"
        length: Length scale in km
        mapping: Mapping type (for unified method)
        optimizer: Optimizer type (for unified method)

    Returns:
        Path to output directory
    """
    if method == "unified":
        return _run_unified(monkeypatch, mapping, optimizer, length)
    elif method == "emulated":
        return _run_emulated(monkeypatch, length)
    elif method == "solved":
        return _run_solved(monkeypatch, length)
    else:
        raise ValueError(f"Unknown method: {method}")


def _run_unified(
    monkeypatch: pytest.MonkeyPatch,
    mapping: str = "identity",
    optimizer: str = "adam",
    length: Optional[int] = None,
) -> str:
    """Run unified method simulation."""
    experiment_optimizer = "lbfgs" if optimizer == "cg_newton" else optimizer
    argv = [
        "igm_run.py",
        f"+experiment=params_{experiment_optimizer}",
        "processes.iceflow.method=unified",
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
        path_run_dir = os.path.join(
            "outputs", f"{length}km", "unified", mapping, optimizer
        )
    else:
        path_run_dir = os.path.join("outputs", "unified", mapping, optimizer)

    argv.append(f"hydra.run.dir={path_run_dir}")

    if mapping == "identity":
        argv.append("processes.iceflow.unified.adam.lr_init=0.9")

    monkeypatch.setattr(sys, "argv", argv)
    main()

    return path_run_dir


def _run_emulated(
    monkeypatch: pytest.MonkeyPatch,
    length: Optional[int] = None,
) -> str:
    """Run emulated method simulation."""
    # TODO: Implement emulated method
    raise NotImplementedError("Emulated method not yet implemented")


def _run_solved(
    monkeypatch: pytest.MonkeyPatch,
    length: Optional[int] = None,
) -> str:
    """Run solved method simulation."""
    # TODO: Implement solved method
    raise NotImplementedError("Solved method not yet implemented")
