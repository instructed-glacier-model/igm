#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def misfit_thkprior(cfg, state):
    """Pull thk toward the thickness prior (thkinit) within a spatially
    variable tolerance.

    sigma(x,y) = thkprior_scale * max(thkinit_err, thkprior_std_floor)

    thkinit_err is the prior's own per-pixel error map. Cells without prior
    data should carry a huge thkinit_err (set by the preprocessing) so the
    prior is inactive there. Where dense observations exist (radar misfit_thk,
    velocities), their much tighter std dominates and the prior only prevents
    the unconstrained thinning of unobserved glaciers.
    """
    fit = cfg.assimilations.data_assimilation.fitting

    sigma = fit.thkprior_scale * tf.maximum(state.thkinit_err, fit.thkprior_std_floor)

    ACT = state.icemaskobs > 0.5

    return 0.5 * tf.reduce_mean(
        ((state.thkinit[ACT] - state.thk[ACT]) / sigma[ACT]) ** 2
    )
