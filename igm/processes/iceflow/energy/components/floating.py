#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag2
from igm.processes.iceflow.utils.vertical_discretization import compute_levels

from abc import ABC, abstractmethod
from typing import Tuple
class EnergyComponent(ABC):
	@abstractmethod
	def cost():
		pass

class FloatingComponent(EnergyComponent):
    def __init__(self, params):
        self.params = params
    # ! Get rid of this dependency on vert_disc if floating does not use it...
    def cost(self, U, V, fieldin, vert_disc, staggered_grid):
        return cost_floating(
            U, V, fieldin, vert_disc, staggered_grid, self.params
        )

class FloatingEnergyParams(tf.experimental.ExtensionType):
    """Floating parameters for the cost function."""
    Nz: int
    vert_spacing: float
    cf_eswn: Tuple[str, ...]
    vert_basis: str

def cost_floating(U, V, fieldin, vert_disc, staggered_grid, floating_params): # ! Update to new signature

    thk, usurf, _, _, dX = fieldin
    
    Nz = floating_params.Nz
    vert_spacing = floating_params.vert_spacing
    cf_eswn = floating_params.cf_eswn
    vert_basis = floating_params.vert_basis

    return _cost(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn, staggered_grid, vert_basis)


@tf.function()
def _cost(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn, staggered_grid, vert_basis):

    if not staggered_grid:
        raise ValueError("Floating cost function requires staggered grid, non-staggered grid is not implmented yet.")      

    if vert_basis == "Legendre": 
        raise ValueError("Floating cost function requires Lagrange or SIA vert_basis, Legendre is not implmented yet.")  

    # if activae this applies the stress condition along the calving front

    lsurf = usurf - thk
    
#   Check formula (17) in [Jouvet and Graeser 2012], Unit is Mpa 
    P =tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ) , 0.0)  / dX[:, 0, 0] 
    
    if len(cf_eswn) == 0:
        thkext = tf.pad(thk,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
        lsurfext = tf.pad(lsurf,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
    else:
        thkext = thk
        thkext = tf.pad(thkext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in cf_eswn))
        thkext = tf.pad(thkext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in cf_eswn))
        thkext = tf.pad(thkext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in cf_eswn))
        thkext = tf.pad(thkext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in cf_eswn)) 
        lsurfext = lsurf
        lsurfext = tf.pad(lsurfext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in cf_eswn))
        lsurfext = tf.pad(lsurfext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in cf_eswn))
        lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in cf_eswn))
        lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in cf_eswn)) 
    
    # this permits to locate the calving front in a cell in the 4 directions
    CF_W = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,:-2]==0)&(lsurfext[:,1:-1,:-2]<=0),1.0,0.0)
    CF_E = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,2:]==0)&(lsurfext[:,1:-1,2:]<=0),1.0,0.0) 
    CF_S = tf.where((lsurf<0)&(thk>0)&(thkext[:,:-2,1:-1]==0)&(lsurfext[:,:-2,1:-1]<=0),1.0,0.0)
    CF_N = tf.where((lsurf<0)&(thk>0)&(thkext[:,2:,1:-1]==0)&(lsurfext[:,2:,1:-1]<=0),1.0,0.0)

    if Nz > 1:
        # Blatter-Pattyn
        levels = compute_levels(Nz, vert_spacing)
        temd = levels[1:] - levels[:-1] 
        weight = tf.stack([tf.ones_like(thk) * z for z in temd], axis=1) # dimensionless, 
        C_float = (
                P * tf.reduce_sum(weight * stag2(U), axis=1) * CF_W  # Check is stag2 is OK !!!
            - P * tf.reduce_sum(weight * stag2(U), axis=1) * CF_E   # Check is stag2 is OK !!!
            + P * tf.reduce_sum(weight * stag2(V), axis=1) * CF_S   # Check is stag2 is OK !!!
            - P * tf.reduce_sum(weight * stag2(V), axis=1) * CF_N  # Check is stag2 is OK !!! 
        ) 
    else:
        # SSA
        C_float = ( P * U * CF_W - P * U * CF_E  + P * V * CF_S - P * V * CF_N )  

    return C_float
