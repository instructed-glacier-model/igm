
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def regu_arrhenius(cfg,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.arrhenius[:, 1:] - state.arrhenius[:, :-1])/state.dx
    dady = (state.arrhenius[1:, :] - state.arrhenius[:-1, :])/state.dx

    if cfg.processes.data_assimilation.optimization.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)
    
    if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
        REGU_S = (cfg.processes.data_assimilation.regularization.arrhenius) * 0.5 * (
            tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
        )
    else:
        REGU_S = (cfg.processes.data_assimilation.regularization.arrhenius) * (
            tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
        )

    return REGU_S