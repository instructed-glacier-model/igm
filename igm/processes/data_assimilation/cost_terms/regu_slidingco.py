#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np


def regu_slidingco(cfg, state):
    """Regularization on the friction control (slidingco or tau_ref).

    The control name is resolved at DA init (see resolve_friction_name in
    data_assimilation/utils.py) and stashed on state.da_friction.
    The matching weight is read from
    cfg.processes.data_assimilation.regularization[<control_name>].
    """
    fric = state.da_friction
    field = getattr(state, fric)
    weight = cfg.processes.data_assimilation.regularization[fric]

    dadx = (field[:, 1:] - field[:, :-1]) / state.dx
    dady = (field[1:, :] - field[:-1, :]) / state.dx

    if cfg.processes.data_assimilation.optimization.sole_mask:
        dadx = tf.where((state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1), dadx, 0.0)
        dady = tf.where((state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1), dady, 0.0)

    if cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor_sl == 1:
        if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
            REGU_S = weight * 0.5 * (
                tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
            )
        else:
            REGU_S = weight * (
                tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
            )
    else:
        dadx = (field[:, 1:] - field[:, :-1]) / state.dx
        dadx = (dadx[1:, :] + dadx[:-1, :]) / 2.0
        dady = (field[1:, :] - field[:-1, :]) / state.dx
        dady = (dady[:, 1:] + dady[:, :-1]) / 2.0

        if cfg.processes.data_assimilation.optimization.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dadx = tf.where(MASK, dadx, 0.0)
            dady = tf.where(MASK, dady, 0.0)

        if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
            REGU_S = weight * 0.5 * (
                (1.0 / np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor_sl))
                * tf.math.reduce_mean((dadx * state.flowdirx + dady * state.flowdiry)**2)
                + np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor_sl)
                * tf.math.reduce_mean((dadx * state.flowdiry - dady * state.flowdirx)**2)
            )
        else:
            REGU_S = weight * (
                (1.0 / np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor_sl))
                * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
                + np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor_sl)
                * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx)))

    return REGU_S
