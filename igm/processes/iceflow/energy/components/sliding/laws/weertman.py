#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict, Tuple

from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.utils.velocities import get_velbase

from ..sliding import SlidingComponent


class WeertmanParams(tf.experimental.ExtensionType):

    exponent: float
    regu_weertman: float
    vert_basis: str


class Weertman(SlidingComponent):

    def __init__(self, params: WeertmanParams):
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict,
        vert_disc: Tuple,
        staggered_grid: bool,
    ) -> tf.Tensor:
        return cost_weertman(U, V, fieldin, vert_disc, staggered_grid, self.params)


def cost_weertman(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: Tuple,
    staggered_grid: bool,
    weertman_params: WeertmanParams,
) -> tf.Tensor:

    thk, usurf, slidingco, dX = (
        fieldin["thk"],
        fieldin["usurf"],
        fieldin["slidingco"],
        fieldin["dX"],
    )
    zeta, dzeta, _, _ = vert_disc

    expo = weertman_params.exponent
    regu = weertman_params.regu_weertman
    vert_basis = weertman_params.vert_basis

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
    staggered_grid,
    vert_basis,
):

    # Coefficient and effective exponent
    C = 1.0 * slidingco
    s = 1.0 + 1.0 / expo

    # Bed gradients
    grad_top_x, grad_top_y = compute_gradient(usurf - thk, dX, dX, staggered_grid)

    # Optional staggering
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    # Retrieve basal velocity
    u_basal, v_basal = get_velbase(U, V, vert_basis)

    # Compute basal velocity magnitude (with norm M and regularization)
    corr_bed = u_basal * grad_top_x + v_basal * grad_top_y

    velbase_mag = tf.sqrt(
        u_basal * u_basal + v_basal * v_basal + regu * regu + corr_bed * corr_bed
    )

    return C * tf.pow(velbase_mag, s) / s
