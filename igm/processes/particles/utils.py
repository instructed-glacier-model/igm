#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

# def zeta_to_rhs(cfg, zeta):
#     return (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
#         1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
#     )

# get the position in the column
def rhs_to_zeta(vert_spacing, rhs):
    if vert_spacing == 1:
        zeta = rhs
    else:
        DET = tf.sqrt(1 + 4 * (vert_spacing - 1) * vert_spacing * rhs)
        zeta = (DET - 1) / (2 * (vert_spacing - 1))
 
    return zeta

