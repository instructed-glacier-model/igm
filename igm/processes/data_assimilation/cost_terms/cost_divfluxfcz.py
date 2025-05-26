#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from scipy import stats 
from igm.utils.gradient.compute_divflux import compute_divflux

def cost_divfluxfcz(cfg,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=cfg.processes.data_assimilation.divflux.method
    )
 
    ACT = state.icemaskobs > 0.5
    if i % 10 == 0:
        # his does not need to be comptued any iteration as this is expensive
        state.res = stats.linregress(
            state.usurf[ACT], divflux[ACT]
        )  # this is a linear regression (usually that's enough)
    # or you may go for polynomial fit (more gl, but may leads to errors)
    #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
    divfluxtar = tf.where(
        ACT, state.res.intercept + state.res.slope * state.usurf, 0.0
    )
#   divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )
    
    ACT = state.icemaskobs > 0.5
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - divflux[ACT]) / cfg.processes.data_assimilation.fitting.divfluxobs_std) ** 2
    )

    if cfg.processes.data_assimilation.divflux.force_zero_sum:
            ACT = state.icemaskobs > 0.5
            COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / cfg.processes.data_assimilation.fitting.divfluxobs_std) ** 2

    return COST_D
 