#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag2
from igm.processes.iceflow.vert_disc import compute_levels


def cost_floating(cfg, U, V, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc
    
    Nz = cfg.processes.iceflow.numerics.Nz
    vert_spacing = cfg.processes.iceflow.numerics.vert_spacing
    cf_eswn = cfg.processes.iceflow.physics.cf_eswn
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_floating(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn, staggered_grid, vert_basis)


@tf.function()
def _cost_floating(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn, staggered_grid, vert_basis):

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

        ###########################################################

        # ddz = tf.stack([thk * z for z in temd], axis=1) 

        # zzz = tf.expand_dims(lsurf, axis=1) + tf.math.cumsum(ddz, axis=1)

        # f = 10 ** (-6) * ( 910 * 9.81 * (tf.expand_dims(usurf, axis=1) - zzz) + 1000 * 9.81 * tf.minimum(0.0, zzz) )  # Mpa m^(-1) 

        # C_float = (
        #       tf.reduce_sum(ddz * f * stag2(U), axis=1) * CF_W
        #     - tf.reduce_sum(ddz * f * stag2(U), axis=1) * CF_E 
        #     + tf.reduce_sum(ddz * f * stag2(V), axis=1) * CF_S 
        #     - tf.reduce_sum(ddz * f * stag2(V), axis=1) * CF_N 
        # )   # Mpa m / y
        
        ##########################################################

        # f = 10 ** (-6) * ( 910 * 9.81 * thk + 1000 * 9.81 * tf.minimum(0.0, lsurf) ) # Mpa 

 
        # sloptopgx, sloptopgy = compute_gradient_tf(lsurf[0], dX[0, 0, 0], dX[0, 0, 0])
        # slopn = (sloptopgx**2 + sloptopgy**2 + 1.e-10 )**0.5
        # nx = tf.expand_dims(sloptopgx/slopn,0)
        # ny = tf.expand_dims(sloptopgy/slopn,0)
            
        # C_float_2 = - tf.where( (thk>0)&(slidingco==0), - f * (U[:,0] * nx + V[:,0] * ny), 0.0 ) # Mpa m/y

        # #C_float is unit  Mpa m * (m/y) / m + Mpa m / y = Mpa m / y    
        # C_float = C_float + C_float_2 
         

    # print(C_shear[0].numpy(),C_slid[0].numpy(),C_grav[0].numpy(),C_float[0].numpy())

    # C_pen = 10000 * tf.where(thk>0,0.0, tf.reduce_sum( tf.abs(U), axis=1)**2 )

    return C_float
