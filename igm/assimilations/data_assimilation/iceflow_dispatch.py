#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Entry points of data_assimilation into the iceflow module. Every call that
data_assimilation makes into iceflow goes through the two helpers below.
"""

import tensorflow as tf

from igm.processes.iceflow.evaluator.evaluator import evaluate_iceflow
from igm.processes.iceflow.solver.solver import solve_iceflow


def iceflow_evaluate(cfg, state):
    """Run the iceflow forward model (inference) to compute velocities."""

    evaluate_iceflow(cfg, state)


def iceflow_retrain(cfg, state):
    """Retrain / re-solve the iceflow model to enforce physics.

    Runs solve_iceflow then evaluate_iceflow, and returns state.cost[-1].
    """

    solve_iceflow(cfg, state)
    evaluate_iceflow(cfg, state)
    # Return a tensor in all cases (a frozen / offline emulator does no
    # retraining, so state.cost may be empty): keep cost["glen"] the same
    # type as the other cost terms so the cost printers (which call
    # .numpy()) work for both the on-line and off-line iceflow.
    if hasattr(state, "cost") and len(state.cost) > 0:
        return state.cost[-1]
    else:
        return tf.constant(0.0)
