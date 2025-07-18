#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf       
from igm.processes.particles.utils import get_weights, get_weights_legendre
 
def update_particles(cfg, state):

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:
        
        if cfg.processes.particles.seeding_method == "accumulation":
            from igm.processes.particles.seeding_particles \
                import seeding_particles_accumulation as seeding_particles
        elif cfg.processes.particles.seeding_method == "all":
            from igm.processes.particles.seeding_particles \
                import seeding_particles_all as seeding_particles

        seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        state.particle["x"] = tf.concat([state.particle["x"], state.nparticle["x"]], axis=-1)
        state.particle["y"] = tf.concat([state.particle["y"], state.nparticle["y"]], axis=-1)
        state.particle["z"] = tf.concat([state.particle["z"], state.nparticle["z"]], axis=-1)
        state.particle["r"] = tf.concat([state.particle["r"], state.nparticle["r"]], axis=-1)
        state.particle["thk"] = tf.concat([state.particle["thk"], state.nparticle["thk"]], axis=-1)
        state.particle["topg"] = tf.concat([state.particle["topg"], state.nparticle["topg"]], axis=-1)
        state.particle["t"] = tf.concat([state.particle["t"], state.nparticle["t"]], axis=-1)

        id = 0 if state.particle["id"].shape[0] == 0 else state.particle["id"][-1]
        state.nparticle["id"] = tf.range(id, id + state.particle["x"].shape[0])
        state.particle["id"] = tf.concat([state.particle["id"], state.nparticle["id"]], axis=-1)

        if "weights" in cfg.processes.particles.fields:
            state.particle["w"] = tf.concat([state.particle["w"], state.nparticle["w"]], axis=-1)
        if "englt" in cfg.processes.particles.fields:
            state.particle["englt"] = tf.concat([state.particle["englt"], state.nparticle["englt"]], axis=-1)
        if "velmag" in cfg.processes.particles.fields:
            state.particle["velmag"] = tf.concat([state.particle["velmag"], state.nparticle["velmag"]], axis=-1)

        state.tlast_seeding = state.t.numpy()

    if (state.particle["x"].shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories, these indicies are real values to permit 
        # 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle["x"]) / state.dx
        j = (state.particle["y"]) / state.dx

        indices = tf.stack([j, i], axis=-1)[tf.newaxis, ...]

        if cfg.processes.particles.computation_library == "cuda":
            from igm.processes.particles.utils_cuda import interpolate_particles_2d       
        elif cfg.processes.particles.computation_library == "cupy":
            from igm.processes.particles.utils_cupy import interpolate_particles_2d       
        elif cfg.processes.particles.computation_library == "tensorflow":
            from igm.processes.particles.utils_tf import interpolate_particles_2d

        u, v, w, smb, state.particle["thk"], state.particle["topg"] = \
            interpolate_particles_2d(state.U, state.V, state.W, state.smb, state.thk, state.topg, indices)
         
        if cfg.processes.iceflow.numerics.vert_basis in ["Lagrange","SIA"]:
            weights = get_weights(
                vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
                Nz=cfg.processes.iceflow.numerics.Nz,
                particle_r=state.particle["r"]
            )
        elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
            weights = get_weights_legendre(state.particle["r"],cfg.processes.iceflow.numerics.Nz)

        state.particle["x"] += state.dt * tf.reduce_sum(weights * u, axis=0)
        state.particle["y"] += state.dt * tf.reduce_sum(weights * v, axis=0)

        state.particle["t"] += state.dt

        if cfg.processes.particles.tracking_method == "simple":

            # adjust the relative height within the ice column with smb
            pudt = state.particle["r"] * (state.particle["thk"] - smb * state.dt) / state.particle["thk"]
            state.particle["r"] = tf.where(state.particle["thk"] > 0.1, tf.clip_by_value(pudt, 0, 1), 1)
            state.particle["z"] = state.particle["topg"] + state.particle["thk"] * state.particle["r"]

        elif cfg.processes.particles.tracking_method == "3d":

            state.particle["z"] += state.dt * tf.reduce_sum(weights * w, axis=0)
            # make sure the particle vertically remain within the ice body
            state.particle["z"] = tf.clip_by_value(state.particle["z"], state.particle["topg"], 
                                                state.particle["topg"] + state.particle["thk"])
            # relative height of the particle within the glacier
            state.particle["r"] = (state.particle["z"] - state.particle["topg"]) / state.particle["thk"]
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle["r"] = tf.where(state.particle["thk"] == 0,
                                        tf.ones_like(state.particle["r"]), state.particle["r"])

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle["x"] = tf.clip_by_value(state.particle["x"], 0, state.x[-1] - state.x[0])
        state.particle["y"] = tf.clip_by_value(state.particle["y"], 0, state.y[-1] - state.y[0])

        if "weights" in cfg.processes.particles.fields:
            indices = tf.stack([j, i], axis=-1)
            updates = tf.where(state.particle["r"] == 1, state.particle["w"], 0.0)
            # this computes the sum of the weight of particles on a 2D grid
            state.weight_particles = tf.tensor_scatter_nd_add(
                tf.zeros_like(state.thk), tf.cast(indices, tf.int32), updates
            )

        if "englt" in cfg.processes.particles.fields:
            state.particle["englt"] += tf.where(state.particle["r"] < 1, state.dt, 0.0)  # englacial time

        if "velmag" in cfg.processes.particles.fields:
            if 'w' in locals():
                state.particle["velmag"] = tf.reduce_sum(weights * tf.sqrt(u**2 + v**2 + w**2), axis=0)
            else:
                state.particle["velmag"] = tf.reduce_sum(weights * tf.sqrt(u**2 + v**2), axis=0)

        # remove particles that are on the bedrock and have negative smb

        # I = tf.where((state.particle_r==1) & (smb < 0))

        # state.particle_id = state.particle_id[I]
        # state.particle_x = state.particle_x[I]
        # state.particle_y = state.particle_y[I]
        # state.particle_z = state.particle_z[I]
        # state.particle_r = state.particle_r[I]
        # state.particle_thk = state.particle_thk[I]
        # state.particle_topg = state.particle_topg[I]
        # state.particle_t = state.particle_t[I]
        # if "weights" in cfg.processes.particles.fields:
        #     state.particle_w = state.particle_w[I]
        # if "englt" in cfg.processes.particles.fields:
        #     state.particle_englt = state.particle_englt[I]
        # if "velmag" in cfg.processes.particles.fields:
        #     state.particle_velmag = state.particle_velmag[I]



        
