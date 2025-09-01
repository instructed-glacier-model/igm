#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf

from igm.utils.math.getmag import getmag
from igm.processes.iceflow.utils import get_velbase

def initialize(cfg, state):
    
    if "time" not in cfg.processes:
        raise ValueError("The 'time' module is required for the 'glerosion' module.")
    
    state.tlast_erosion = tf.Variable(cfg.processes.time.start, dtype=tf.float32)


def update(cfg, state):
    if (state.t - state.tlast_erosion) >= cfg.processes.glerosion.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "update topg_glacial_erosion at time : " + str(state.t.numpy())
            )

        velbase_mag = getmag(*get_velbase(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis))

        # apply erosion law, erosion rate is proportional to a power of basal sliding speed
        dtopgdt = cfg.processes.glerosion.cst * (velbase_mag**cfg.processes.glerosion.exp)

        state.topg = state.topg - (state.t - state.tlast_erosion) * dtopgdt

        # THIS WORK ONLY FOR GROUNDED ICE, TO BE ADAPTED FOR FLOATING ICE
        state.usurf = state.topg + state.thk

        state.tlast_erosion.assign(state.t)


def finalize(cfg, state):
    pass
