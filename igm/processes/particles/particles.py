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
from igm.processes.particles.update_particles import update_particles

# def srange(message, color):
#     tf.test.experimental.sync_devices()
#     return nvtx.start_range(message, color)

# def erange(rng):
#     tf.test.experimental.sync_devices()
#     nvtx.end_range(rng)

def initialize(cfg, state):

    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required to use the particles module")
    
    if cfg.processes.particles.tracking.method == "3d":
        if "vert_flow" not in cfg.processes:
            raise ValueError(
                "The 'vert_flow' module is required to use the 3d tracking method in the 'particles' module."
            )

    state.tlast_seeding = cfg.processes.particles.seeding.tlast_init

    state.particle = {}  # this is a dictionary to store the particles
    state.nparticle = {}  # this is a dictionary to store the new particles

    # initialize trajectoriesstate.nparticle["weight"]
    state.particle["id"] = tf.Variable([],dtype=tf.int32)
 
    for key in ["x", "y", "z", "r", "t"]:
        state.particle[key] = tf.Variable([])

    for key in cfg.processes.particles.output.add_fields:
        state.particle[key] = tf.Variable([])

    # build the gridseed, we don't want to seed all pixels!
    state.gridseed = np.zeros_like(state.thk) == 1
    # uniform seeding on the grid
    rr = int(1.0 / cfg.processes.particles.seeding.density)
    state.gridseed[::rr, ::rr] = True

    if cfg.processes.particles.output.library == "numpy":
        initialize_write_particle_numpy(cfg, state)
    elif cfg.processes.particles.output.library == "cudf":
        initialize_write_particle_cudf(cfg, state)

def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    update_particles(cfg, state)

#        rng = srange("Writing particles", color="blue")
    if cfg.processes.particles.output.library == "numpy":
        update_write_particle_numpy(cfg, state)
    elif cfg.processes.particles.output.library == "cudf":
        update_write_particle_cudf(cfg, state)
#        erange(rng)
        
def finalize(cfg, state):
    pass

