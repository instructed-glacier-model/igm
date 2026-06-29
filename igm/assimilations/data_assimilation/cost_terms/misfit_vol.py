#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def misfit_vol(cfg,state):

    voltar = tf.reduce_sum(state.thkinit)

    volmod = tf.reduce_sum(state.thk)
 
    return 0.5 * ( ( voltar - volmod ) / (voltar * cfg.assimilations.data_assimilation.fitting.volobsprop_std ) )** 2