#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

# from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.utils.grad.compute_divflux import compute_divflux
from ..cost_terms.total_cost import total_cost

from ..iceflow_dispatch import iceflow_evaluate

from ..utils import compute_flow_direction_for_anisotropic_smoothing_vel


def optimize_update_lbfgs(cfg, state, cost, i):

    try:
        import tensorflow_probability as tfp
    except ImportError:
        raise ImportError(
            "The 'lbfgs' optimizer requires tensorflow-probability. "
            "Install it with: pip install tensorflow-probability"
        )

    sc = {}
    sc["thk"] = cfg.processes.data_assimilation.scaling.thk
    sc["usurf"] = cfg.processes.data_assimilation.scaling.usurf
    sc[state.da_friction] = cfg.processes.data_assimilation.scaling[state.da_friction]
    sc["arrhenius"] = cfg.processes.data_assimilation.scaling.arrhenius

    for f in cfg.processes.data_assimilation.control_list:
        setattr(state, f + "_sc", tf.Variable(getattr(state, f) / sc[f]))
        if cfg.processes.data_assimilation.fitting.log_slidingco & (f == state.da_friction):
            setattr(state, f + "_sc", tf.Variable(tf.sqrt(getattr(state, f) / sc[f])))
        else:
            setattr(state, f + "_sc", tf.Variable(getattr(state, f) / sc[f]))

    Cost_Glen = []

    def COST(controls):

        cost = {}

        for i, f in enumerate(cfg.processes.data_assimilation.control_list):
            setattr(state, f + "_sc", controls[i])

        for f in cfg.processes.data_assimilation.control_list:
            if cfg.processes.data_assimilation.fitting.log_slidingco & (
                f == state.da_friction
            ):
                setattr(state, f, (getattr(state, f + "_sc") ** 2) * sc[f])
            else:
                setattr(state, f, getattr(state, f + "_sc") * sc[f])

        iceflow_evaluate(cfg, state)

        if (
            not cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor
            == 1
        ):
            compute_flow_direction_for_anisotropic_smoothing_vel(state)

        return total_cost(cfg, state, cost, i)

    def loss_and_gradients_function(controls):
        with tf.GradientTape() as tape:
            tape.watch(controls)
            cost = COST(controls)
            gradients = tape.gradient(cost, controls)
        return cost, gradients

    controls = tf.stack(
        [getattr(state, f + "_sc") for f in cfg.processes.data_assimilation.control_list],
        axis=0,
    )

    optimizer = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=loss_and_gradients_function,
        initial_position=controls,
        max_iterations=cfg.processes.iceflow.solver.nbitmax,
        tolerance=1e-8,
    )

    controls = optimizer.position

    for i, f in enumerate(cfg.processes.data_assimilation.control_list):
        setattr(state, f + "_sc", controls[i])

    state.divflux = compute_divflux(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        method=cfg.processes.data_assimilation.divflux.method,
    )
