#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.utils.grad.compute_divflux import compute_divflux

def cost_divfluxpen(cfg, state, i):
    """Pure smoothness penalty on the flux divergence (no target).

    COST = regularization.divflux * 0.5 * mean(|grad divflux|^2)

    Unlike cost_divfluxfcz, this does not pull divflux toward any prescribed
    (SMB-like, linear-in-elevation) shape: it only penalizes the grid-scale
    roughness of divflux, letting the optimizer decide what the smooth pattern
    should be. Rationale: a real glacier always has a smooth flux divergence
    (divflux = smb - dh/dt, both smooth and bounded), so roughness of divflux
    is a proxy for physical inconsistency of the (thk, velocity) pair.
    """

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx,
        method=cfg.assimilations.data_assimilation.divflux.method,
        smooth_sigma=cfg.assimilations.data_assimilation.divflux.smooth_sigma
    )

    dddx = (divflux[:, 1:] - divflux[:, :-1]) / state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :]) / state.dx

    return (
        cfg.assimilations.data_assimilation.regularization.divflux
        * 0.5
        * (tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2))
    )
