#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import tensorflow as tf

from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
# from igm.processes.iceflow.utils.misc import get_velbase

from igm.processes.iceflow.utils.velocities import get_velbase

class EnergyComponent(ABC):
	@abstractmethod
	def cost():
		pass

class SlidingWeertmanComponent(EnergyComponent):
    def __init__(self, params):
        self.params = params
    def cost(self, U, V, fieldin, vert_disc, staggered_grid):
        return cost_sliding_weertman(
            U, V, fieldin, vert_disc, staggered_grid, self.params
        )

class SlidingWeertmanEnergyParams(tf.experimental.ExtensionType):
    """Sliding Weertman parameters for the cost function."""
    exp_weertman: float
    regu_weertman: float
    vert_basis: str

# ! I dont think this is needed if we have a sliding_law argument - its a bit confusing...
def cost_sliding_weertman(U: tf.Tensor, V: tf.Tensor, fieldin: Dict, vert_disc: Tuple, staggered_grid: bool, sliding_weertman_params: SlidingWeertmanEnergyParams):

    thk, usurf, slidingco, dX = fieldin["thk"], fieldin["usurf"], fieldin["slidingco"], fieldin["dX"]
    zeta, dzeta = vert_disc

    exp_weertman = sliding_weertman_params.exp_weertman
    regu_weertman = sliding_weertman_params.regu_weertman
    vert_basis = sliding_weertman_params.vert_basis

    return _cost(U, V, thk, usurf, slidingco, dX, zeta, dzeta,
                         exp_weertman, regu_weertman, staggered_grid, vert_basis)

@tf.function()
def _cost(U, V, thk, usurf, slidingco, dX, zeta, dzeta, \
                  exp_weertman, regu_weertman, staggered_grid, vert_basis):
 
    C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
 
    s = 1.0 + 1.0 / exp_weertman
  
    sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 

    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    uvelbase, vvelbase = get_velbase(U, V, vert_basis)

    N = ( (uvelbase ** 2 + vvelbase ** 2) + regu_weertman**2 \
        + (uvelbase * sloptopgx + vvelbase * sloptopgy) ** 2 )

    return C * N ** (s / 2) / s # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y