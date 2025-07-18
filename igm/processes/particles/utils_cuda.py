
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf      
import math

import os
import igm

igm_path = os.path.dirname(igm.__file__)
particles_path = os.path.join(igm_path, "processes", "particles")
interpolate_op = tf.load_op_library(os.path.join(particles_path, 'interpolate_2d', 'interpolate_2d.so'))
 
def interpolate_particles_2d(U, V, W, SMB, THK, TOPG, indices):

    # Make depth = 1 since the interpolate2d function expects a 3D tensor
    thk_input = tf.expand_dims(THK, axis=0)
    topg_input = tf.expand_dims(TOPG, axis=0)
    smb_input = tf.expand_dims(SMB, axis=0)
    
    particles_tf = tf.squeeze(indices)
    u = interpolate_op.interpolate2d(grid=U, particles=particles_tf)
    v = interpolate_op.interpolate2d(grid=V, particles=particles_tf)
    w = interpolate_op.interpolate2d(grid=W, particles=particles_tf)
    smb = interpolate_op.interpolate2d(grid=smb_input, particles=particles_tf)
    thk = interpolate_op.interpolate2d(grid=thk_input, particles=particles_tf)
    topg = interpolate_op.interpolate2d(grid=topg_input, particles=particles_tf)

    # Remove the extra dimension added for depth
    smb = tf.squeeze(smb)
    thk = tf.squeeze(thk)
    topg = tf.squeeze(topg)

    return u, v, w, smb, thk, topg