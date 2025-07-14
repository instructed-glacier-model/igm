#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf       

from igm.processes.particles.seeding_particles import seeding_particles
from igm.processes.particles.utils import rhs_to_zeta

import os
import igm

igm_path = os.path.dirname(igm.__file__)
particles_path = os.path.join(igm_path, "processes", "particles")
interpolate_op = tf.load_op_library(os.path.join(particles_path, 'interpolate_2d', 'interpolate_2d.so'))

def get_weights(vert_spacing, number_z_layers, particle_r, u):
    "What is this function doing? Name it properly.."

    # rng_outer = srange("indices in weights", color="blue")
    zeta = rhs_to_zeta(vert_spacing, particle_r)  # get the position in the column
    I0 = tf.math.floor(zeta * (number_z_layers - 1))

    I0 = tf.minimum(I0, number_z_layers - 2)  # make sure to not reach the upper-most pt
    I1 = I0 + 1

    zeta0 = I0 / (number_z_layers - 1)
    zeta1 = I1 / (number_z_layers - 1)
    lamb = (zeta - zeta0) / (zeta1 - zeta0)

    ind0 = tf.stack([tf.cast(I0, tf.int64), tf.range(I0.shape[0], dtype=tf.int64)], axis=1)
    ind1 = tf.stack([tf.cast(I1, tf.int64), tf.range(I1.shape[0], dtype=tf.int64)], axis=1)
    
    weights = tf.zeros_like(u)
    weights = tf.tensor_scatter_nd_add(
        weights, indices=ind0, updates=1 - lamb
    )
    weights = tf.tensor_scatter_nd_add(
        weights, indices=ind1, updates=lamb
    )

    return weights
 
def update_cuda(cfg, state):

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:

        seeding_particles(cfg, state)

        nparticle_x = state.nparticle_x
        nparticle_y = state.nparticle_y
        nparticle_z = state.nparticle_z
        nparticle_r = state.nparticle_r
        nparticle_w = state.nparticle_w
        nparticle_t = state.nparticle_t
        nparticle_englt = state.nparticle_englt

        # # seed new particles
        # (
        #     nparticle_x,
        #     nparticle_y,
        #     nparticle_z,
        #     nparticle_r,
        #     nparticle_w,
        #     nparticle_t,
        #     nparticle_englt,
        # ) = seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        particle_x = tf.Variable(
            tf.concat([state.particle_x, nparticle_x], axis=-1), trainable=False
        )
        particle_y = tf.Variable(
            tf.concat([state.particle_y, nparticle_y], axis=-1), trainable=False
        )
        particle_z = tf.Variable(
            tf.concat([state.particle_z, nparticle_z], axis=-1), trainable=False
        )
        state.particle_r = tf.Variable(
            tf.concat([state.particle_r, nparticle_r], axis=-1), trainable=False
        )
        state.particle_w = tf.Variable(
            tf.concat([state.particle_w, nparticle_w], axis=-1), trainable=False
        )
        state.particle_t = tf.Variable(
            tf.concat([state.particle_t, nparticle_t], axis=-1), trainable=False
        )
        particle_englt = tf.Variable(
            tf.concat([state.particle_englt, nparticle_englt], axis=-1),
            trainable=False,
        )

        state.tlast_seeding = state.t.numpy()
    else:
        # use the old particles
        particle_x = state.particle_x
        particle_y = state.particle_y
        particle_z = state.particle_z
        particle_englt = state.particle_englt

    if (particle_x.shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (particle_x) / state.dx
        j = (particle_y) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        U_input = state.U
        V_input = state.V
        W_input = state.W
        thk_input = state.thk
        topg_input = state.topg
        # smb_input = state.smb

        
        # Make depth = 1 since the interpolate2d function expects a 3D tensor
        thk_input = tf.expand_dims(thk_input, axis=0)
        topg_input = tf.expand_dims(topg_input, axis=0)
        
        particles_tf = tf.squeeze(indices)
        u = interpolate_op.interpolate2d(grid=U_input, particles=particles_tf)
        v = interpolate_op.interpolate2d(grid=V_input, particles=particles_tf)
        w = interpolate_op.interpolate2d(grid=W_input, particles=particles_tf)
        thk = interpolate_op.interpolate2d(grid=thk_input, particles=particles_tf)
        topg = interpolate_op.interpolate2d(grid=topg_input, particles=particles_tf)

        # Remove the extra dimension added for depth
        thk = tf.squeeze(thk)
        topg = tf.squeeze(topg)
        
        state.particle_thk = thk
        state.particle_topg = topg

        weights = get_weights(
            vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
            number_z_layers=cfg.processes.iceflow.numerics.Nz,
            particle_r=state.particle_r,
            u=u,
        )

        particle_x = particle_x + state.dt * tf.reduce_sum(weights * u, axis=0)
        particle_y = particle_y + state.dt * tf.reduce_sum(weights * v, axis=0)
        particle_z = particle_z + state.dt * tf.reduce_sum(weights * w, axis=0)

        # make sure the particle vertically remain within the ice body
        state.particle_z = tf.clip_by_value(particle_z, topg, topg + thk)
        # relative height of the particle within the glacier
        particle_r = (state.particle_z - topg) / thk
        # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
        state.particle_r = tf.where(thk == 0, tf.ones_like(particle_r), particle_r)

        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(particle_x, 0, state.x[-1] - state.x[0])
        state.particle_y = tf.clip_by_value(particle_y, 0, state.y[-1] - state.y[0])

        indices = tf.concat(
            [
                tf.expand_dims(j, axis=-1),
                tf.expand_dims(i, axis=-1),
            ],
            axis=-1,
        )
        updates = tf.where(state.particle_r == 1, state.particle_w, 0.0)

        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), tf.cast(indices, tf.int32), updates
        )

        # compute the englacial time
        state.particle_englt = particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

