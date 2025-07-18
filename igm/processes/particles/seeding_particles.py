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
    # here we seed where i) thickness is higher than 1 m 
    #                    ii) on the gridseed (which permit to control the seeding density)
    #                    iii) on the accumulation area
    
    I = (state.thk > 1) & state.gridseed & (state.smb > 0) # here you may redefine how you want to seed particles

    state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle_z = state.usurf[I]           # z position of the particle
    state.nparticle_thk = state.thk[I]           # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]         # z position of the bedrock under the particle 

    state.nparticle_t = tf.ones_like(state.nparticle_x) * state.t 
    state.nparticle_r = (state.nparticle_z - state.nparticle_topg) / state.nparticle_thk
    state.nparticle_r = tf.where(state.nparticle_thk == 0, tf.ones_like(state.nparticle_r), state.nparticle_r)

    if "weights" in cfg.processes.particles.fields:
        state.nparticle_w = tf.ones_like(state.nparticle_x)
    if "englt" in cfg.processes.particles.fields:
        state.nparticle_englt = tf.zeros_like(state.nparticle_x)
    if "velmag" in cfg.processes.particles.fields:
        state.nparticle_velmag = tf.zeros_like(state.nparticle_x)