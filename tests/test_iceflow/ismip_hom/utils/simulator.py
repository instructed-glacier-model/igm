#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import sys
import tensorflow as tf
from igm.igm_run import main
from typing import Optional

RANDOM_SEED = 0
tf.config.experimental.enable_op_determinism()


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
    argv = [
        "igm_run.py",
        f"+experiment=params_{optimizer}",
        "processes.iceflow.method=unified",
        f"processes.iceflow.unified.mapping={mapping}",
        f"processes.iceflow.unified.optimizer={optimizer}",
    ]

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
    tf.random.set_seed(RANDOM_SEED)
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
