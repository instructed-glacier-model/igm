#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

def seeding_particles_accumulation(cfg, state):
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

    state.nparticle["x"] = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle["y"] = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle["z"] = state.usurf[I]           # z position of the particle
    state.nparticle["thk"] = state.thk[I]           # ice thickness at position of the particle
    state.nparticle["topg"] = state.topg[I]         # z position of the bedrock under the particle

    state.nparticle["t"] = tf.ones_like(state.nparticle["x"]) * state.t
    state.nparticle["r"] = (state.nparticle["z"] - state.nparticle["topg"]) / state.nparticle["thk"]
    state.nparticle["r"] = tf.where(state.nparticle["thk"] == 0, tf.ones_like(state.nparticle["r"]), state.nparticle["r"])

    if "weight" in cfg.processes.particles.fields:
        state.nparticle["weight"] = tf.ones_like(state.nparticle["x"])
    if "englt" in cfg.processes.particles.fields:
        state.nparticle["englt"] = tf.zeros_like(state.nparticle["x"])
    if "velmag" in cfg.processes.particles.fields:
        state.nparticle["velmag"] = tf.zeros_like(state.nparticle["x"])

def seeding_particles_all(cfg, state):
    """
    User seeding particles in the glacier
    """

    np_x = []
    np_y = []
    np_z = []
    np_y = []
    np_thk = []
    np_topg = []

    dz = 20 

    for i in np.arange(0, int(tf.reduce_max(state.thk)/dz)):

        ice_thk = i*dz
        I = (ice_thk<state.thk) & state.gridseed

        # Skip appending if I is entirely False
        if not tf.reduce_any(I):
            continue

        np_x.append(state.X[I] - state.x[0])  # x position of the particle
        np_y.append(state.Y[I] - state.y[0])  # y position of the particle
        np_z.append(state.topg[I] + ice_thk)
        np_thk.append(state.thk[I])  # ice thickness at position of the particle
        np_topg.append(state.topg[I])  # z position of the bedrock under the particle

    state.nparticle["x"] = tf.concat(np_x, axis=0)
    state.nparticle["y"] = tf.concat(np_y, axis=0)
    state.nparticle["z"] = tf.concat(np_z, axis=0)
    state.nparticle["thk"] = tf.concat(np_thk, axis=0)
    state.nparticle["topg"] = tf.concat(np_topg, axis=0)

    state.nparticle["t"] = tf.ones_like(state.nparticle["x"]) * state.t
    state.nparticle["r"] = (state.nparticle["z"] - state.nparticle["topg"]) / state.nparticle["thk"]
    state.nparticle["r"] = tf.where(state.nparticle["thk"] == 0, tf.ones_like(state.nparticle["r"]), state.nparticle["r"])

    if "weight" in cfg.processes.particles.fields:
        state.nparticle["weight"] = tf.ones_like(state.nparticle["x"])
    if "englt" in cfg.processes.particles.fields:
        state.nparticle["englt"] = tf.zeros_like(state.nparticle["x"])
    if "velmag" in cfg.processes.particles.fields:
        state.nparticle["velmag"] = tf.zeros_like(state.nparticle["x"])
