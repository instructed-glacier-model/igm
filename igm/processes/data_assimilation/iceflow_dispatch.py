#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Dispatch layer for data_assimilation to work with both the 'emulated' and
'unified' iceflow methods.  Every call that data_assimilation makes into the
iceflow module goes through the two helpers below so that the rest of the
data_assimilation code is iceflow-method-agnostic.
"""


def _get_method(cfg):
    return cfg.processes.iceflow.method.lower()


def iceflow_evaluate(cfg, state):
    """Run iceflow forward model (inference) to compute velocities.

    Emulated: calls update_iceflow_emulated
    Unified:  calls evaluate_iceflow
    """
    method = _get_method(cfg)

    if method == "emulated":
        from igm.processes.iceflow.emulate.emulated import update_iceflow_emulated

        update_iceflow_emulated(cfg, state)

    elif method == "unified":
        from igm.processes.iceflow.unified.evaluator.evaluator import evaluate_iceflow

        evaluate_iceflow(cfg, state)

    else:
        raise ValueError(
            f"data_assimilation does not support iceflow method '{method}'. "
            "Use 'emulated' or 'unified'."
        )


def iceflow_retrain(cfg, state):
    """Retrain / re-solve the iceflow model to enforce physics.

    Emulated: retrains the emulator CNN then runs inference.
              Returns state.cost_emulator[-1].
    Unified:  runs solve_iceflow then evaluate_iceflow.
              Returns state.cost (scalar).
    """
    method = _get_method(cfg)

    if method == "emulated":
        from igm.processes.iceflow.emulate.emulator import update_iceflow_emulator

        update_iceflow_emulator(cfg, state)
        return state.cost_emulator[-1]

    elif method == "unified":
        from igm.processes.iceflow.unified.solver.solver import solve_iceflow
        from igm.processes.iceflow.unified.evaluator.evaluator import evaluate_iceflow

        solve_iceflow(cfg, state)
        evaluate_iceflow(cfg, state)
        return state.cost[-1] if hasattr(state, "cost") and len(state.cost) > 0 else 0.0

    else:
        raise ValueError(
            f"data_assimilation does not support iceflow method '{method}'. "
            "Use 'emulated' or 'unified'."
        )
