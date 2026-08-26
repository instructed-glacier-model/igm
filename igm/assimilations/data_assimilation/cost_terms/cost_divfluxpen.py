#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from ..utils import compute_forward_divflux

def cost_divfluxpen(cfg, state, i):
    """Pure smoothness penalty on the flux divergence (no target).

    COST = regularization.divflux * 0.5 * mean(|grad divflux|^2)

    Unlike cost_divfluxfcz, this does not pull divflux toward any prescribed
    (SMB-like, linear-in-elevation) shape: it only penalizes the grid-scale
    roughness of divflux, letting the optimizer decide what the smooth pattern
    should be. Rationale: a real glacier always has a smooth flux divergence
    (divflux = smb - dh/dt, both smooth and bounded), so roughness of divflux
    is a proxy for physical inconsistency of the (thk, velocity) pair.

    The divergence is ALWAYS the forward transport operator's one
    (compute_forward_divflux — single source of truth in this module):
    smoothness is operator-dependent, and penalizing the operator that will
    actually advance the ice makes "smooth divflux in the inversion" mean,
    by definition, "no shock at forward start". It is also RAW by
    construction (no filtering): penalizing a filtered divergence leaves the
    sub-filter modes of the controls unpenalized and the optimizer fills
    that blind spot with grid noise (Aletsch 2026-07 post-mortem: apparent
    grain 0.2 m/yr, true raw grain 29-40 m/yr over a ~3900-trial search).
    """

    divflux = compute_forward_divflux(cfg, state)

    dddx = (divflux[:, 1:] - divflux[:, :-1]) / state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :]) / state.dx

    # Mask-aware smoothness, as regu_thk and regu_slidingco already do. Taken
    # across the outline, the gradient sees the step from the interior divflux
    # to 0 outside the ice; flattening that step costs less flux, which thins
    # the margin. Default false keeps the historical behaviour.
    if cfg.assimilations.data_assimilation.regularization.get("divflux_sole_mask", False):
        mx = (state.icemaskobs[:, 1:] > 0.5) & (state.icemaskobs[:, :-1] > 0.5)
        my = (state.icemaskobs[1:, :] > 0.5) & (state.icemaskobs[:-1, :] > 0.5)
        dddx = tf.where(mx, dddx, 0.0)
        dddy = tf.where(my, dddy, 0.0)

    return (
        cfg.assimilations.data_assimilation.regularization.divflux
        * 0.5
        * (tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2))
    )
