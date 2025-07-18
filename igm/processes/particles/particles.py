#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf
# import nvtx

from igm.processes.particles.write_particle_numpy import initialize_write_particle_numpy 
from igm.processes.particles.write_particle_numpy import update_write_particle_numpy
from igm.processes.particles.write_particle_cudf import initialize_write_particle_cudf
from igm.processes.particles.write_particle_cudf import update_write_particle_cudf 
from igm.processes.particles.write_particle_pyvista import initialize_write_particle_pyvista
from igm.processes.particles.write_particle_pyvista import update_write_particle_pyvista 
from igm.processes.particles.update_particles import update_particles

# def srange(message, color):
#     tf.test.experimental.sync_devices()
#     return nvtx.start_range(message, color)

# def erange(rng):
#     tf.test.experimental.sync_devices()
#     nvtx.end_range(rng)

def initialize(cfg, state):

    if cfg.processes.particles.tracking_method == "3d":
        if "vert_flow" not in cfg.processes:
            raise ValueError(
                "The 'vert_flow' module is required to use the 3d tracking method in the 'particles' module."
            )

    state.tlast_seeding = cfg.processes.particles.tlast_seeding_init

    state.particle = {}  # this is a dictionary to store the particles
    state.nparticle = {}  # this is a dictionary to store the new particles

    # initialize trajectories
    state.particle["id"] = tf.Variable([],dtype=tf.int32)
    state.particle["x"] = tf.Variable([])
    state.particle["y"] = tf.Variable([])
    state.particle["z"] = tf.Variable([])
    state.particle["r"] = tf.Variable([])
    state.particle["topg"] = tf.Variable([])
    state.particle["thk"] = tf.Variable([]) 
    state.particle["t"] = tf.Variable([])

    if "weights" in cfg.processes.particles.fields:
        state.particle["w"] = tf.Variable([])  # this is to give a weight to the particle
    if "englt" in cfg.processes.particles.fields:
        state.particle["englt"] = tf.Variable([])
    if "velmag" in cfg.processes.particles.fields:
        state.particle["velmag"] = tf.Variable([])

    # build the gridseed, we don't want to seed all pixels!
    state.gridseed = np.zeros_like(state.thk) == 1
    # uniform seeding on the grid
    rr = int(1.0 / cfg.processes.particles.density_seeding)
    state.gridseed[::rr, ::rr] = True

    if cfg.processes.particles.write_trajectories:
        if cfg.processes.particles.writing_library == "numpy":
            initialize_write_particle_numpy(cfg, state)
        elif cfg.processes.particles.writing_library == "cudf":
            initialize_write_particle_cudf(cfg, state)
        elif cfg.processes.particles.writing_library == "pyvista":
            initialize_write_particle_pyvista(cfg, state)
        else:
            raise ValueError("must be either 'numpy' or 'cudf'.")

def update(cfg, state):

    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required to use the particles module")

    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    update_particles(cfg, state)

    if cfg.processes.particles.write_trajectories:
#        rng = srange("Writing particles", color="blue")
        if cfg.processes.particles.writing_library == "numpy":
            update_write_particle_numpy(cfg, state)
        elif cfg.processes.particles.writing_library == "cudf":
            update_write_particle_cudf(cfg, state)
        elif cfg.processes.particles.writing_library == "pyvista":
            update_write_particle_pyvista(cfg, state)
        else:
            raise ValueError("Must be either 'numpy' or 'cudf'.")
#        erange(rng)
        
def finalize(cfg, state):
    pass

