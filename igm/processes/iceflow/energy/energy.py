#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.energy.utils import gauss_points_and_weights, stag4
from igm.processes.iceflow.vert_disc import compute_levels, compute_dz
from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 


from igm.processes.iceflow.energy.cost_gravity_2layers import cost_gravity_2layers
from igm.processes.iceflow.energy.cost_shear_2layers import cost_shear_2layers

from igm.processes.iceflow.energy.cost_shear import cost_shear
from igm.processes.iceflow.energy.cost_sliding import cost_sliding
from igm.processes.iceflow.energy.cost_gravity import cost_gravity
from igm.processes.iceflow.energy.cost_floating import cost_floating

def iceflow_energy(cfg, U, V, fieldin):
    thk, usurf, arrhenius, slidingco, dX = fieldin

    # In that case, we assume the iceflow has a SIA-like profile
    if cfg.processes.iceflow.numerics.Nz == 2:

        exp_glen = cfg.processes.iceflow.physics.exp_glen
        exp_weertman = cfg.processes.iceflow.physics.exp_weertman
        regu_glen = cfg.processes.iceflow.physics.regu_glen
        regu_weertman = cfg.processes.iceflow.physics.regu_weertman
        ice_density = cfg.processes.iceflow.physics.ice_density
        gravity_cst = cfg.processes.iceflow.physics.gravity_cst
        new_friction_param = cfg.processes.iceflow.physics.new_friction_param
 
        n, w = gauss_points_and_weights(ord_gauss=3)
  
        Cshear = cost_shear_2layers(thk, arrhenius, U, V, dX, exp_glen, regu_glen, w, n)
 
        Cslid =  cost_sliding(U, V, thk, usurf, slidingco, dX, 
                              exp_weertman, regu_weertman, new_friction_param)

        Cgrav = cost_gravity_2layers(U, V, thk, usurf, dX, exp_glen, ice_density, gravity_cst, w, n)

        Cfloat = tf.zeros_like(Cshear) # not implemented for 2 layers

        return Cshear, Cslid, Cgrav, Cfloat
 
    # In that case, it can be SSA if Nz=1 or Blaater-Pattyn if Nz>2
    else:

        Nz = cfg.processes.iceflow.numerics.Nz
        vert_spacing = cfg.processes.iceflow.numerics.vert_spacing
        exp_glen = cfg.processes.iceflow.physics.exp_glen
        exp_weertman = cfg.processes.iceflow.physics.exp_weertman
        regu_glen = cfg.processes.iceflow.physics.regu_glen
        regu_weertman = cfg.processes.iceflow.physics.regu_weertman
        thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
        ice_density = cfg.processes.iceflow.physics.ice_density
        gravity_cst = cfg.processes.iceflow.physics.gravity_cst
        new_friction_param = cfg.processes.iceflow.physics.new_friction_param
        cf_cond = cfg.processes.iceflow.physics.cf_cond
        cf_eswn = cfg.processes.iceflow.physics.cf_eswn
        regu = cfg.processes.iceflow.physics.regu
        min_sr = cfg.processes.iceflow.physics.min_sr
        max_sr = cfg.processes.iceflow.physics.max_sr
        force_negative_gravitational_energy = cfg.processes.iceflow.physics.force_negative_gravitational_energy
 
        levels = compute_levels(Nz, vert_spacing)

        dz =  compute_dz(stag4(thk), levels)

        COND = ( (thk[:, 1:, 1:] > 0) & (thk[:, 1:, :-1] > 0)
               & (thk[:, :-1, 1:] > 0) & (thk[:, :-1, :-1] > 0) )
        COND = tf.expand_dims(COND, axis=1)

        Cshear = cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, dz, COND, 
                            exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, regu)

        Cslid =  cost_sliding(U, V, thk, usurf, slidingco, dX, 
                              exp_weertman, regu_weertman, new_friction_param)

        Cgrav = cost_gravity(U, V, usurf, dX, dz, COND, Nz, ice_density, gravity_cst, 
                               force_negative_gravitational_energy)
        
        if cf_cond:
            Cfloat = cost_floating(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn)
        else:
            Cfloat = tf.zeros_like(Cshear)

        return Cshear, Cslid, Cgrav, Cfloat
     
def iceflow_energy_XY(cfg, X, Y):
    
    U, V = Y_to_UV(cfg, Y)

    fieldin = X_to_fieldin(cfg, X)

    return iceflow_energy(cfg, U, V, fieldin)
