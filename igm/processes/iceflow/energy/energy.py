#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.vert_disc import compute_levels, compute_zeta_dzeta
from igm.processes.iceflow.energy.utils import gauss_points_and_weights
from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(cfg, U, V, fieldin):

    thk, usurf, arrhenius, slidingco, dX = fieldin
 
    levels = compute_levels(cfg.processes.iceflow.numerics.Nz, 
                            cfg.processes.iceflow.numerics.vert_spacing)
 
    if cfg.processes.iceflow.numerics.vert_basis == "Lagrange":
        zeta, dzeta = compute_zeta_dzeta(levels)
    elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
        zeta, dzeta = gauss_points_and_weights(ord_gauss=cfg.processes.iceflow.numerics.Nz)
    elif cfg.processes.iceflow.numerics.vert_basis == "SIA":
        assert cfg.processes.iceflow.numerics.Nz == 2 # Only works in this case
        zeta, dzeta = gauss_points_and_weights(ord_gauss=5)
    else:
        raise ValueError(f"Unknown vertical basis: {cfg.processes.iceflow.numerics.vert_basis}")

    energy_list = []
    for component in cfg.processes.iceflow.physics.energy_components:
        func = getattr(energy, f"cost_{component}")
        energy_list.append(func(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta))

    return energy_list

def iceflow_energy_XY(cfg, X, Y):
    
    U, V = Y_to_UV(cfg, Y)

    fieldin = X_to_fieldin(cfg, X)

    return iceflow_energy(cfg, U, V, fieldin)
