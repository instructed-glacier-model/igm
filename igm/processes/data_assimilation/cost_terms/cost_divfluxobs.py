#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.utils.gradient.compute_divflux import compute_divflux

def cost_divfluxobs(cfg,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=cfg.processes.data_assimilation.divflux.method
    )
 
    divfluxtar = state.divfluxobs
    ACT = ~tf.math.is_nan(divfluxtar)
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - divflux[ACT]) / cfg.processes.data_assimilation.fitting.divfluxobs_std) ** 2
    )
 
    dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
    COST_D += (cfg.processes.data_assimilation.regularization.divflux) * 0.5 * ( tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2) )

    if cfg.processes.data_assimilation.divflux.force_zero_sum:
        ACT = state.icemaskobs > 0.5
        COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / cfg.processes.data_assimilation.fitting.divfluxobs_std) ** 2

    return COST_D