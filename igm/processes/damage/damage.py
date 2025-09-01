#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import tensorflow as tf
import os, igm
from igm.processes.particles.utils import get_weights_lagrange
from igm.processes.particles.utils_interp import scatter_to_3d_grid
 
def initialize(cfg, state):
 
    if "stress" not in cfg.processes:
        raise ValueError("The 'stress' module is required for the 'damage' module.")

    if "particles" not in cfg.processes:
        raise ValueError("The 'particles' module is required for the 'damage' module.")
    
    state.particle['damage'] = tf.Variable([])
    
def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update DAMAGE at time : " + str(state.t.numpy()))
 
    # this is to ensure that the damage variable is initialized with zeros for new entries.
    pad_len = tf.shape(state.particle['x'])[0] - tf.shape(state.particle['damage'])[0]
    if pad_len > 0:
        state.particle['damage'] = tf.pad(state.particle['damage'], [[0, pad_len]], constant_values=0)
 
    B = cfg.processes.damage.B
    r = cfg.processes.damage.r
    k = cfg.processes.damage.k
    epsilon = cfg.processes.damage.epsilon
    alpha = cfg.processes.damage.alpha
    beta = cfg.processes.damage.beta
    sigmabar = cfg.processes.damage.sigmabar
    lamb = cfg.processes.damage.lamb

    # interpolation of stress fields to particles
    igm_path = os.path.dirname(igm.__file__)
    particles_path = os.path.join(igm_path, "processes", "particles")
    interpolate_op = tf.load_op_library(os.path.join(particles_path, 'interpolate_2d', 'interpolate_2d.so'))
    i = state.particle["x"] / state.dx
    j = state.particle["y"] / state.dx
    indices = tf.stack([j, i], axis=-1)
    sigma1 = interpolate_op.interpolate2d(grid=state.sigma1, particles=indices)
    tauII = interpolate_op.interpolate2d(grid=state.tauII, particles=indices)
    sigmaI = interpolate_op.interpolate2d(grid=state.sigmaI, particles=indices)

    weights = get_weights_lagrange(
        vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
        Nz=cfg.processes.iceflow.numerics.Nz,
        particle_r=state.particle["r"]
    )

    sigma1 = tf.reduce_sum(sigma1 * weights, axis=0)
    tauII  = tf.reduce_sum(tauII * weights, axis=0)
    sigmaI = tf.reduce_sum(sigmaI * weights, axis=0)

    # Compute psi
    term = (alpha * sigma1 + beta * tf.sqrt(3 * tauII) + (1 - alpha - beta) * sigmaI) # Mpa
    psi = 1.0e+6 * term / (1 - state.particle['damage']) - (1.0e+6 * sigmabar)        # Pa

    # Compute source term f
    positive = tf.maximum(0, psi) ** r
    negative = tf.maximum(0, -psi) ** r
    f = B * (positive - lamb * negative) / (1 - state.particle['damage'])**k
 
    # Explicit time integration of D
    state.particle['damage'] += state.dt * f

    # Ensure D stays in [0, 1-epsilon]
    state.particle['damage'] = tf.clip_by_value(state.particle['damage'], 0.0, 1 - epsilon)

    # Scatter damage to 3D grid
    state.damage = scatter_to_3d_grid(state.particle, "damage", state.dx, state.U.shape, 
                                      state.levels, cfg.processes.iceflow.numerics.vert_spacing)
    
    # Ensure D stays in [0, 1-epsilon]
    state.damage = tf.clip_by_value(state.damage, 0.0, 1 - epsilon)

    # Update arrhenius field based on damage
    state.arrhenius /= (1 - state.damage)**cfg.processes.iceflow.physics.exp_glen
    
def finalize(cfg, state):
    pass
