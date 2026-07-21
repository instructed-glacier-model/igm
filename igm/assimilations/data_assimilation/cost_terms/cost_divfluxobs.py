#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from ..utils import compute_forward_divflux

def cost_divfluxobs(cfg,state,i):

    # Fit AND regularize the SAME flux divergence that grain_div measures and the
    # forward model advances (forward transport operator), not the upwind one:
    # smoothness is operator-dependent, so fitting/penalizing upwind left the
    # forward-operator divflux rough (grain blew up on fast trunks). Aligns
    # divfluxobs with divfluxpen (203ff65).
    divflux = compute_forward_divflux(cfg, state)
 
    divfluxtar = state.divfluxobs
    ACT = ~tf.math.is_nan(divfluxtar)
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - divflux[ACT]) / cfg.assimilations.data_assimilation.fitting.divfluxobs_std) ** 2
    )
 
    dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
    COST_D += (cfg.assimilations.data_assimilation.regularization.divflux) * 0.5 * ( tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2) )

    if cfg.assimilations.data_assimilation.divflux.force_zero_sum:
        ACT = state.icemaskobs > 0.5
        COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / cfg.assimilations.data_assimilation.fitting.divfluxobs_std) ** 2

    return COST_D