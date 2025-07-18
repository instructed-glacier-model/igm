#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf       

from igm.processes.particles.seeding_particles import seeding_particles
from igm.processes.particles.utils import get_weights, get_weights_legendre
 
def update_particles(cfg, state):

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:
        
        seeding_particles(cfg, state)            

        # merge the new seeding points with the former ones
        state.particle_x = tf.concat([state.particle_x, state.nparticle_x], axis=-1)
        state.particle_y = tf.concat([state.particle_y, state.nparticle_y], axis=-1)
        state.particle_z = tf.concat([state.particle_z, state.nparticle_z], axis=-1)
        state.particle_r = tf.concat([state.particle_r, state.nparticle_r], axis=-1)
        state.particle_thk = tf.concat([state.particle_thk, state.nparticle_thk], axis=-1)
        state.particle_topg = tf.concat([state.particle_topg, state.nparticle_topg], axis=-1)
        state.particle_t = tf.concat([state.particle_t, state.nparticle_t], axis=-1)

        if "weights" in cfg.processes.particles.fields:
            state.particle_w = tf.concat([state.particle_w, state.nparticle_w], axis=-1)
        if "englt" in cfg.processes.particles.fields:
            state.particle_englt = tf.concat([state.particle_englt, state.nparticle_englt], axis=-1)            
        if "velmag" in cfg.processes.particles.fields:
            state.particle_velmag = tf.concat([state.particle_velmag, state.nparticle_velmag], axis=-1)
 
        state.tlast_seeding = state.t.numpy()

    if (state.particle_x.shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories, these indicies are real values to permit 
        # 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx

        indices = tf.stack([j, i], axis=-1)[tf.newaxis, ...]

        if cfg.processes.particles.tracking_method == "3d":
            if cfg.processes.particles.computation_library == "cuda":
                from igm.processes.particles.utils_cuda import interpolate_particles_2d       
            elif cfg.processes.particles.computation_library == "cupy":
                from igm.processes.particles.utils_cupy import interpolate_particles_2d       
            elif cfg.processes.particles.computation_library == "tensorflow":
                from igm.processes.particles.utils_tf import interpolate_particles_2d
            
            u, v, w, state.particle_thk, state.particle_topg = \
                interpolate_particles_2d(state.U, state.V, state.W, state.thk, state.topg, indices)

        elif cfg.processes.particles.tracking_method == "simple":
            if cfg.processes.particles.computation_library == "cuda":
                from igm.processes.particles.utils_cuda import interpolate_particles_2d_simple
            elif cfg.processes.particles.computation_library == "cupy":
                print("The cupy computation library is not implemented for the simple tracking method.")
            elif cfg.processes.particles.computation_library == "tensorflow":
                from igm.processes.particles.utils_tf import interpolate_particles_2d_simple

            u, v, smb, state.particle_thk, state.particle_topg = \
                interpolate_particles_2d_simple(state.U, state.V, state.smb, state.thk, state.topg, indices)
         
        if cfg.processes.iceflow.numerics.vert_basis in ["Lagrange","SIA"]:
            weights = get_weights(
                vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
                Nz=cfg.processes.iceflow.numerics.Nz,
                particle_r=state.particle_r
            )
        elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
            weights = get_weights_legendre(state.particle_r,cfg.processes.iceflow.numerics.Nz)

        state.particle_x += state.dt * tf.reduce_sum(weights * u, axis=0)
        state.particle_y += state.dt * tf.reduce_sum(weights * v, axis=0)

        state.particle_t += state.dt 

        if cfg.processes.particles.tracking_method == "simple":

            # adjust the relative height within the ice column with smb
            pudt = state.particle_r * (state.particle_thk - smb * state.dt) / state.particle_thk
            state.particle_r = tf.where(state.particle_thk > 0.1, tf.clip_by_value(pudt, 0, 1), 1)
            state.particle_z = state.particle_topg + state.particle_thk * state.particle_r

        elif cfg.processes.particles.tracking_method == "3d":

            state.particle_z += state.dt * tf.reduce_sum(weights * w, axis=0)
            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, state.particle_topg, 
                                                state.particle_topg + state.particle_thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - state.particle_topg) / state.particle_thk
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle_r = tf.where(state.particle_thk == 0, 
                                        tf.ones_like(state.particle_r), state.particle_r)

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(state.particle_x, 0, state.x[-1] - state.x[0])
        state.particle_y = tf.clip_by_value(state.particle_y, 0, state.y[-1] - state.y[0])

        if "weights" in cfg.processes.particles.fields:
            indices = tf.stack([j, i], axis=-1)
            updates = tf.where(state.particle_r == 1, state.particle_w, 0.0)
            # this computes the sum of the weight of particles on a 2D grid
            state.weight_particles = tf.tensor_scatter_nd_add(
                tf.zeros_like(state.thk), tf.cast(indices, tf.int32), updates
            )

        if "englt" in cfg.processes.particles.fields:
            state.particle_englt += tf.where(state.particle_r < 1, state.dt, 0.0)  # englacial time

        if "velmag" in cfg.processes.particles.fields:
            if 'w' in locals():
                state.particle_velmag = tf.reduce_sum(weights * tf.sqrt(u**2 + v**2 + w**2), axis=0)
            else:
                state.particle_velmag = tf.reduce_sum(weights * tf.sqrt(u**2 + v**2), axis=0)
