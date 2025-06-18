#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf 

from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf
from igm.processes.particles.seeding_particles_tf import seeding_particles
from igm.processes.particles.utils import rhs_to_zeta

def update_tf(cfg, state):

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:

        seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        state.particle_x = tf.Variable(
            tf.concat([state.particle_x, state.nparticle_x], axis=-1), trainable=False
        )
        state.particle_y = tf.Variable(
            tf.concat([state.particle_y, state.nparticle_y], axis=-1), trainable=False
        )
        state.particle_z = tf.Variable(
            tf.concat([state.particle_z, state.nparticle_z], axis=-1), trainable=False
        )
        state.particle_r = tf.Variable(
            tf.concat([state.particle_r, state.nparticle_r], axis=-1), trainable=False
        )
        state.particle_w = tf.Variable(
            tf.concat([state.particle_w, state.nparticle_w], axis=-1), trainable=False
        )
        state.particle_t = tf.Variable(
            tf.concat([state.particle_t, state.nparticle_t], axis=-1), trainable=False
        )
        state.particle_englt = tf.Variable(
            tf.concat([state.particle_englt, state.nparticle_englt], axis=-1),
            trainable=False,
        )
        state.particle_topg = tf.Variable(
            tf.concat([state.particle_topg, state.nparticle_topg], axis=-1),
            trainable=False,
        )
        state.particle_thk = tf.Variable(
            tf.concat([state.particle_thk, state.nparticle_thk], axis=-1),
            trainable=False,
        )

        state.tlast_seeding = state.t.numpy()

    if (state.particle_x.shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        u = interpolate_bilinear_tf(
            tf.expand_dims(state.U, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        v = interpolate_bilinear_tf(
            tf.expand_dims(state.V, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        thk = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_thk = thk

        topg = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_topg = topg

        smb = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        zeta = rhs_to_zeta(cfg, state.particle_r)  # get the position in the column
        I0 = tf.cast(
            tf.math.floor(zeta * (cfg.processes.iceflow.numerics.Nz - 1)),
            dtype="int32",
        )
        I0 = tf.minimum(
            I0, cfg.processes.iceflow.numerics.Nz - 2
        )  # make sure to not reach the upper-most pt
        I1 = I0 + 1
        zeta0 = tf.cast(I0 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")
        zeta1 = tf.cast(I1 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")

        lamb = (zeta - zeta0) / (zeta1 - zeta0)

        ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
        ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))

        wei = tf.zeros_like(u)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

        if cfg.processes.particles.tracking_method == "simple":
            # adjust the relative height within the ice column with smb
            state.particle_r = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle_r * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = topg + thk * state.particle_r

        elif cfg.processes.particles.tracking_method == "3d":
            # uses the vertical velocity w computed in the vert_flow module

            w = interpolate_bilinear_tf(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = state.particle_z + state.dt * tf.reduce_sum(
                wei * w, axis=0
            )

            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - topg) / thk
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle_r = tf.where(
                thk == 0, tf.ones_like(state.particle_r), state.particle_r
            )

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(
            state.particle_x, 0, state.x[-1] - state.x[0]
        )
        state.particle_y = tf.clip_by_value(
            state.particle_y, 0, state.y[-1] - state.y[0]
        )

        indices = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(
            tf.where(state.particle_r == 1, state.particle_w, 0), dtype="float32"
        )

        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), indices, updates
        )

        # compute the englacial time
        state.particle_englt = state.particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        #    if int(state.t)%10==0:
        #        print("nb of part : ",state.xpos.shape)

