#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict, Tuple

from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage
from igm.processes.iceflow.energy.utils import stag4h
from igm.processes.iceflow.utils.velocities import get_velbase
from igm.utils.gradient.compute_gradient import compute_gradient

from ..sliding import SlidingComponent


class CoulombParams(tf.experimental.ExtensionType):

    regu: float
    exponent: float
    mu: float
    vert_basis: str


class Coulomb(SlidingComponent):

    def __init__(self, params: CoulombParams):
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict,
        vert_disc: Tuple,
        staggered_grid: bool,
    ) -> tf.Tensor:
        return cost_coulomb(U, V, fieldin, vert_disc, staggered_grid, self.params)


def cost_coulomb(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: Tuple,
    staggered_grid: bool,
    coulomb_params: CoulombParams,
) -> tf.Tensor:

    thk, usurf, slidingco, dX = (
        fieldin["thk"],
        fieldin["usurf"],
        fieldin["slidingco"],
        fieldin["dX"],
    )
    zeta, dzeta, _, _ = vert_disc

    expo = coulomb_params.exponent
    regu = coulomb_params.regu
    mu = coulomb_params.mu
    vert_basis = coulomb_params.vert_basis

    return _cost(
        U,
        V,
        thk,
        usurf,
        slidingco,
        dX,
        zeta,
        dzeta,
        expo,
        regu,
        mu,
        staggered_grid,
        vert_basis,
    )


@tf.function()
def _cost(
    U,
    V,
    thk,
    usurf,
    slidingco,
    dX,
    zeta,
    dzeta,
    expo,
    regu,
    mu,
    staggered_grid,
    vert_basis,
):
    # Temporary fix for effective pressure - should be within the inputs
    N = get_effective_pressure_precentage(thk, percentage=0.0)
    Nmin = tf.constant(0.25e6)
    N = tf.where(N < Nmin, Nmin, N)

    # Coefficient and effective exponent
    C = 1.0 * slidingco
    s = 1.0 + 1.0 / expo

    # Bed gradients
    dbdx, dbdy = compute_gradient(usurf - thk, dX, dX, staggered_grid)

    # Optional staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    # Retrieve basal velocity
    ux_b, uy_b = get_velbase(U, V, vert_basis)

    # Compute basal velocity magnitude (with norm M and regularization)
    corr_bed = ux_b * dbdx + uy_b * dbdy

    # Compute smooth transition between Weertman and Coulomb following Shapero et al. (2021)
    tau_c = mu * N
    u_c = tf.pow(tau_c / C, expo)
    u_b = tf.sqrt(ux_b * ux_b + uy_b * uy_b + regu * regu + corr_bed * corr_bed)
    tau = tau_c * (tf.pow(tf.pow(u_b, s) + tf.pow(u_c, s), 1.0 / s) - u_c)

    return tf.where(C > 0.0, tau, 0.0)
