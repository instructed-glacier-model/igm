#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import tensorflow as tf

from igm.processes.vert_flow.vert_flow_v1 import compute_vertical_velocity_kinematic_v1, compute_vertical_velocity_incompressibility_v1
from igm.processes.vert_flow.vert_flow_v2 import compute_vertical_velocity_kinematic_v2, compute_vertical_velocity_incompressibility_v2
from igm.processes.vert_flow.vert_flow_v2 import compute_vertical_velocity_twolayers
from igm.processes.vert_flow.vert_flow_legendre import compute_vertical_velocity_legendre   
from igm.processes.iceflow.utils import get_velbase_1, get_velsurf_1

 
def initialize(cfg, state):
    pass

def update(cfg, state):
    """ """

    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    if vert_basis == "Lagrange":

        # original version by GJ
        if cfg.processes.vert_flow.version == 1:

            if cfg.processes.vert_flow.method == "kinematic":
                state.W = compute_vertical_velocity_kinematic_v1(cfg, state)
            elif cfg.processes.vert_flow.method == "incompressibility":
                state.W = compute_vertical_velocity_incompressibility_v1(cfg, state)
    
        # improved version by CMS
        elif cfg.processes.vert_flow.version == 2:

            if cfg.processes.vert_flow.method == "kinematic":
                state.W = compute_vertical_velocity_kinematic_v2(cfg, state)
            elif cfg.processes.vert_flow.method == "incompressibility":
                state.W = compute_vertical_velocity_incompressibility_v2(cfg, state)

    elif vert_basis == "Legendre":
        
        state.W = compute_vertical_velocity_legendre(cfg, state)

    elif vert_basis == "SIA":
        
        state.W = compute_vertical_velocity_twolayers(cfg, state)

    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
  
    state.wvelbase = get_velbase_1(state.W, vert_basis)
    state.wvelsurf = get_velsurf_1(state.W, vert_basis)

def finalize(cfg, state):
    pass
