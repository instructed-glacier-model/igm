#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple

from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia
from igm.utils.gradient.compute_gradient import compute_gradient
from .energy import EnergyComponent


class GravityComponent(EnergyComponent):

    def __init__(self, params):
        self.params = params

    def cost(self, U, V, fieldin, vert_disc, staggered_grid):
        return cost_gravity(U, V, fieldin, vert_disc, staggered_grid, self.params)


class GravityParams(tf.experimental.ExtensionType):

    exp_glen: float
    ice_density: float
    gravity_cst: float
    force_negative_gravitational_energy: bool
    vert_basis: str


def get_gravity_params_args(cfg) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "exp_glen": cfg_physics.exp_glen,
        "ice_density": cfg_physics.ice_density,
        "gravity_cst": cfg_physics.gravity_cst,
        "force_negative_gravitational_energy": cfg_physics.force_negative_gravitational_energy,
        "vert_basis": cfg_numerics.vert_basis,
    }


def cost_gravity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict,
    vert_disc: Tuple,
    staggered_grid: bool,
    gravity_params: GravityParams,
) -> tf.Tensor:

    thk, usurf, dX = fieldin["thk"], fieldin["usurf"], fieldin["dX"]
    zeta, dzeta, Leg_P, _ = vert_disc

    exp_glen = gravity_params.exp_glen
    ice_density = gravity_params.ice_density
    gravity_cst = gravity_params.gravity_cst
    fnge = gravity_params.force_negative_gravitational_energy
    vert_basis = gravity_params.vert_basis

    return _cost(
        U,
        V,
        usurf,
        dX,
        zeta,
        dzeta,
        thk,
        Leg_P,
        ice_density,
        gravity_cst,
        fnge,
        exp_glen,
        staggered_grid,
        vert_basis,
    )


@tf.function()
def _cost(
    U,
    V,
    usurf,
    dX,
    zeta,
    dzeta,
    thk,
    Leg_P,
    ice_density,
    gravity_cst,
    fnge,
    exp_glen,
    staggered_grid,
    vert_basis,
):

    slopsurfx, slopsurfy = compute_gradient(usurf, dX, dX, staggered_grid)

    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        thk = stag4h(thk)

    if vert_basis.lower() == "lagrange":

        U = stag2v(U)
        V = stag2v(V)

    elif vert_basis.lower() == "legendre":

        U = tf.einsum("ij,bjkl->bikl", Leg_P, U)
        V = tf.einsum("ij,bjkl->bikl", Leg_P, V)

    elif vert_basis.lower() == "sia":

        U = U[:, 0:1, :, :] + (U[:, -1:, :, :] - U[:, 0:1, :, :]) * psia(
            zeta[None, :, None, None], exp_glen
        )
        V = V[:, 0:1, :, :] + (V[:, -1:, :, :] - V[:, 0:1, :, :]) * psia(
            zeta[None, :, None, None], exp_glen
        )

    uds = U * slopsurfx[:, None, :, :] + V * slopsurfy[:, None, :, :]

    if fnge:
        uds = tf.minimum(uds, 0.0)  # force non-postiveness

    #    uds = tf.where(thk[:, None, :, :]>0, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    return (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * thk
        * tf.reduce_sum(dzeta[None, :, None, None] * uds, axis=1)
    )
