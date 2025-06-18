#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def zeta_to_rhs(cfg, zeta):
    return (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
        1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
    )

def rhs_to_zeta(cfg, rhs):
    if cfg.processes.iceflow.numerics.vert_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(
            1
            + 4
            * (cfg.processes.iceflow.numerics.vert_spacing - 1)
            * cfg.processes.iceflow.numerics.vert_spacing
            * rhs
        )
        zeta = (DET - 1) / (2 * (cfg.processes.iceflow.numerics.vert_spacing - 1))

    #           temp = cfg.processes.iceflow.numerics.Nz*(DET-1)/(2*(cfg.processes.iceflow.numerics.vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,cfg.processes.iceflow.numerics.Nz-1),dtype='int32')

    return zeta
