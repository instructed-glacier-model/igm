#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def regu_usurf(cfg, state):
    """Smoothness regularization of the surface-elevation control.

    Penalizes the gradient of the DEVIATION (usurf - usurfobs), not of usurf
    itself (which carries the real surface slope):

        COST = regularization.usurf * 0.5 * mean(|grad(usurf - usurfobs)|^2)

    Rationale: when the usurf misfit is relaxed (large usurfobs_std) to give
    the inversion freedom, the optimizer otherwise dumps grid-scale noise into
    the surface (per-cell driving-stress tuning). This term makes the allowed
    deviation from the observed DEM spatially smooth.
    """
    dev = state.usurf - state.usurfobs

    ddx = (dev[:, 1:] - dev[:, :-1]) / state.dx
    ddy = (dev[1:, :] - dev[:-1, :]) / state.dx

    if cfg.assimilations.data_assimilation.optimization.sole_mask:
        ddx = tf.where((state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1), ddx, 0.0)
        ddy = tf.where((state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1), ddy, 0.0)

    return (
        cfg.assimilations.data_assimilation.regularization.usurf
        * 0.5
        * (tf.reduce_mean(ddx**2) + tf.reduce_mean(ddy**2))
    )
