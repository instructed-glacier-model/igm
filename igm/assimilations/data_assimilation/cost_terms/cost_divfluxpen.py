#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.utils.grad.compute_divflux import compute_divflux
from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter

def cost_divfluxpen(cfg, state, i):
    """Pure smoothness penalty on the flux divergence (no target).

    COST = regularization.divflux * 0.5 * mean(|grad divflux|^2)

    Unlike cost_divfluxfcz, this does not pull divflux toward any prescribed
    (SMB-like, linear-in-elevation) shape: it only penalizes the grid-scale
    roughness of divflux, letting the optimizer decide what the smooth pattern
    should be. Rationale: a real glacier always has a smooth flux divergence
    (divflux = smb - dh/dt, both smooth and bounded), so roughness of divflux
    is a proxy for physical inconsistency of the (thk, velocity) pair.

    This penalty acts on a RAW divergence BY CONSTRUCTION (no filtering,
    independent of divflux.method). Penalizing a filtered or centered
    divergence would leave the sub-filter/checkerboard modes of the controls
    unpenalized while they remain useful to the misfit terms — the optimizer
    then systematically fills that blind spot with grid noise (Aletsch
    2026-07: ~3900-trial search, apparent grain 0.2 m/yr but true raw grain
    29-40 m/yr). The conservative flux smoothing remains available for the
    forward model (processes.thk.divflux_smooth_sigma), which faces no
    adversarial optimizer.

    divflux.pen_operator selects WHICH raw divergence is penalized:
      - "upwind":        first-order upwind stencil (historical default).
      - "slope_limiter": the forward model's own transport operator
        (superbee, with a nominal time step divflux.pen_dt). Smoothness of a
        field is operator-dependent: a 50-yr freely-evolved glacier has a
        divflux grain of ~0.2 m/yr under its own transport operator but ~4-7
        under first-order upwind — and inversions penalized with upwind end
        up 20-30x rougher than that natural level when handed to the forward
        model. Penalizing the forward operator makes "smooth divflux in the
        inversion" mean, by definition, "no shock at forward start".
    """

    dfcfg = cfg.assimilations.data_assimilation.divflux
    if dfcfg.pen_operator == "slope_limiter":
        divflux = compute_divflux_slope_limiter(
            state.ubar, state.vbar, state.thk, state.dx, state.dx,
            tf.constant(dfcfg.pen_dt, tf.float32), slope_type="superbee",
        )
    else:
        divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx,
            method="upwind", smooth_sigma=0.0,
        )

    dddx = (divflux[:, 1:] - divflux[:, :-1]) / state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :]) / state.dx

    return (
        cfg.assimilations.data_assimilation.regularization.divflux
        * 0.5
        * (tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2))
    )
