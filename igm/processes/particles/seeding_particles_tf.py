#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

def seeding_particles(cfg, state):
    """
    here we define (xpos,ypos) the horiz coordinate of tracked particles
    and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (a bit more), where there is
    significant ice, and in some points of a regular grid state.gridseed
    (density defined by density_seeding)

    """

    # ! THK and SMB modules are required. Insert in the init function of the particles module (actually, don't because the modules can be
    # ! initialized in any order, and the particles module is not guaranteed to be initialized after the thk and smb modules)
    # ! Instead, insert it HERE when needed (although it might call it multiple times and be less efficient...)

    if not hasattr(state, "thk"):
        raise ValueError("The thk module is required to use the particles module")
    if not hasattr(state, "smb"):
        raise ValueError(
            "A smb module is required to use the particles module. Please use the built-in smb module or create a custom one that overwrites the 'state.smb' value."
        )

    #        This will serve to remove imobile particles, but it is not active yet.

    #        indices = tf.expand_dims( tf.concat(
    #                       [tf.expand_dims((state.ypos - state.y[0]) / state.dx, axis=-1),
    #                        tf.expand_dims((state.xpos - state.x[0]) / state.dx, axis=-1)],
    #                       axis=-1 ), axis=0)

    #        thk = interpolate_bilinear_tf(
    #                    tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
    #                    indices,indexing="ij",      )[0, :, 0]

    #        J = (thk>1)

    # here we seed where i) thickness is higher than 1 m
    #                    ii) the seeding field of geology.nc is active
    #                    iii) on the gridseed (which permit to control the seeding density)
    #                    iv) on the accumulation area
    I = (
        (state.thk > 1) & state.gridseed & (state.smb > 0)
    )  # here you may redefine how you want to seed particles
    state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle_z = state.usurf[I]  # z position of the particle
    state.nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
    state.nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
    state.nparticle_t = (
        tf.ones_like(state.X[I]) * state.t
    )  # "date of birth" of the particle (useful to compute its age)
    state.nparticle_englt = tf.zeros_like(
        state.X[I]
    )  # time spent by the particle burried in the glacier
    state.nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]  # z position of the bedrock under the particle

