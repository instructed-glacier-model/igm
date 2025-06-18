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

    I = (
        (state.thk > 1) & state.gridseed & (state.smb > 0)
    )  # here you may redefine how you want to seed particles
    X_seeded = state.X[I]
    nparticle_x = X_seeded - state.x[0]  # x position of the particle
    nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    nparticle_z = state.usurf[I]  # z position of the particle
    nparticle_r = tf.ones_like(X_seeded)  # relative position in the ice column
    nparticle_w = tf.ones_like(X_seeded)  # weight of the particle
    nparticle_t = (
        tf.ones_like(X_seeded) * state.t
    )  # "date of birth" of the particle (useful to compute its age)
    nparticle_englt = tf.zeros_like(
        X_seeded
    )  # time spent by the particle burried in the glacier
    # nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    # nparticle_topg = state.topg[I]  # z position of the bedrock under the particle

    return (
        nparticle_x,
        nparticle_y,
        nparticle_z,
        nparticle_r,
        nparticle_w,
        nparticle_t,
        nparticle_englt,

    )

