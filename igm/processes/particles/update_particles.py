#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf       
from igm.processes.particles.utils import get_weights_lagrange, get_weights_legendre
from igm.processes.particles.utils_interp import scatter_to_3d_grid
from igm.processes.particles.remove_particles import remove_particles_ablation
 
def update_particles(cfg, state):

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.seeding.frequency:
        
        if not cfg.processes.particles.seeding.method == "user":
            if cfg.processes.particles.seeding.method == "accumulation":
                from igm.processes.particles.seeding_particles \
                    import seeding_particles_accumulation as seeding_particles
            elif cfg.processes.particles.seeding.method == "all":
                from igm.processes.particles.seeding_particles \
                    import seeding_particles_all as seeding_particles
            seeding_particles(cfg, state)
        else:
            seeding_particles_user(cfg, state)

        for key in ["id","x", "y", "z", "r", "t"]:
            state.particle[key] = tf.concat([state.particle[key], state.nparticle[key]], axis=-1)

        for key in cfg.processes.particles.output.add_fields:
            state.particle[key] = tf.concat([state.particle[key], state.nparticle[key]], axis=-1)

        state.tlast_seeding = state.t.numpy()

    if (state.particle["x"].shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories, these indicies are real values to permit 
        # 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle["x"]) / state.dx
        j = (state.particle["y"]) / state.dx

        indices = tf.stack([j, i], axis=-1)[tf.newaxis, ...]

        if cfg.processes.particles.tracking.library == "cuda":
            from igm.processes.particles.utils_cuda import interpolate_particles_2d       
        elif cfg.processes.particles.tracking.library == "cupy":
            from igm.processes.particles.utils_cupy import interpolate_particles_2d       
        elif cfg.processes.particles.tracking.library == "tensorflow":
            from igm.processes.particles.utils_tf import interpolate_particles_2d

        WW = state.W if hasattr(state, 'W') else state.U * 0.0

        u, v, w, smb, thk, topg = \
            interpolate_particles_2d(state.U, state.V, WW, state.smb, state.thk, state.topg, indices)

        if cfg.processes.iceflow.numerics.vert_basis in ["Lagrange","SIA"]:
            weights = get_weights_lagrange(
                vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
                Nz=cfg.processes.iceflow.numerics.Nz,
                particle_r=state.particle["r"]
            )
        elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
            weights = get_weights_legendre(state.particle["r"],cfg.processes.iceflow.numerics.Nz)

        state.particle["x"] += state.dt * tf.reduce_sum(weights * u, axis=0)
        state.particle["y"] += state.dt * tf.reduce_sum(weights * v, axis=0)

        state.particle["t"] += state.dt

        if cfg.processes.particles.tracking.method == "simple":

            # adjust the relative height within the ice column with smb
            pudt = state.particle["r"] * (thk - smb * state.dt) / thk
            state.particle["r"] = tf.where(thk > 0.1, tf.clip_by_value(pudt, 0, 1), 1)
            state.particle["z"] = topg + thk * state.particle["r"]

        elif cfg.processes.particles.tracking.method == "3d":

            state.particle["z"] += state.dt * tf.reduce_sum(weights * w, axis=0)
            # make sure the particle vertically remain within the ice body
            state.particle["z"] = tf.clip_by_value(state.particle["z"], topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle["r"] = (state.particle["z"] - topg) / thk
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle["r"] = tf.where(thk == 0, tf.ones_like(state.particle["r"]), state.particle["r"])

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle["x"] = tf.clip_by_value(state.particle["x"], 0, state.x[-1] - state.x[0])
        state.particle["y"] = tf.clip_by_value(state.particle["y"], 0, state.y[-1] - state.y[0])

        if "weight" in cfg.processes.particles.output.add_fields:
            indices = tf.stack([j, i], axis=-1)
            updates = tf.where(state.particle["r"] == 1, state.particle["weight"], 0.0)
            # this computes the sum of the weight of particles on a 2D grid
            state.weight_particles = tf.tensor_scatter_nd_add(
                tf.zeros_like(state.thk), tf.cast(indices, tf.int32), updates
            )

        if "englt" in cfg.processes.particles.output.add_fields:
            state.particle["englt"] += tf.where(state.particle["r"] < 1, state.dt, 0.0)  # englacial time

        if "velmag" in cfg.processes.particles.output.add_fields:
            state.particle["velmag"] = tf.reduce_sum(weights * tf.sqrt(u**2 + v**2 + w**2), axis=0)
            #state.U2 = scatter_to_3d_grid(state.particle, "velmag", state.dx, state.U.shape, 
            #                              state.levels, cfg.processes.iceflow.numerics.vert_spacing)

    if cfg.processes.particles.removal.method == "ablation":
        remove_particles_ablation(cfg, state)


