#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
 
#from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.utils.gradient.compute_divflux import compute_divflux
from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf
from ..cost_terms.total_cost import total_cost

#from igm.processes.iceflow.emulate.emulate import update_iceflow_emulator, save_iceflow_model, match_fieldin_dimensions
#from igm.processes.iceflow.utils.misc import is_retrain, prepare_data, get_emulator_data

from igm.processes.iceflow.emulate.emulated import update_iceflow_emulated
from igm.processes.iceflow.utils.data_preprocessing import get_fieldin


from ..utils import compute_flow_direction_for_anisotropic_smoothing

def optimize_update(cfg, state, cost, i):

    sc = {}
    sc["thk"] = cfg.processes.data_assimilation.scaling.thk
    sc["usurf"] = cfg.processes.data_assimilation.scaling.usurf
    sc["slidingco"] = cfg.processes.data_assimilation.scaling.slidingco
    sc["arrhenius"] = cfg.processes.data_assimilation.scaling.arrhenius

    for f in cfg.processes.data_assimilation.control_list:
        if cfg.processes.data_assimilation.fitting.log_slidingco & (f == "slidingco"):
            vars(state)[f+'_sc'] = tf.Variable( tf.sqrt(vars(state)[f] / sc[f]) ) 
        else:
            vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f]) 

    with tf.GradientTape() as t:

        if cfg.processes.data_assimilation.optimization.step_size_decay < 1:
            state.optimizer.lr = cfg.processes.data_assimilation.optimization.step_size * (cfg.processes.data_assimilation.optimization.step_size_decay ** (i / 100))

        # is necessary to remember all operation to derive the gradients w.r.t. control variables
        for f in cfg.processes.data_assimilation.control_list:
            t.watch(vars(state)[f+'_sc'])

        for f in cfg.processes.data_assimilation.control_list:
            if cfg.processes.data_assimilation.fitting.log_slidingco & (f == "slidingco"):
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        fieldin = get_fieldin(cfg, state)

        update_iceflow_emulated(cfg, state, fieldin)
        
        if not cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor == 1:
            compute_flow_direction_for_anisotropic_smoothing(state)
                
        cost_total = total_cost(cfg, state, cost, i)

        var_to_opti = [ ]
        for f in cfg.processes.data_assimilation.control_list:
            var_to_opti.append(vars(state)[f+'_sc'])

        # Compute gradient of COST w.r.t. X
        grads = tf.Variable(t.gradient(cost_total, var_to_opti))

        # this serve to restict the optimization of controls to the mask
        if cfg.processes.data_assimilation.optimization.sole_mask:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilation.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                else:
                    grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
        else:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilation.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

        # One step of descent -> this will update input variable X
        state.optimizer.apply_gradients(
            zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
        )

        ###################

        # get back optimized variables in the pool of state.variables
        for f in cfg.processes.data_assimilation.control_list:
            if cfg.processes.data_assimilation.fitting.log_slidingco & (f == "slidingco"):
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        # add reprojection step to force obstacle constraints
        if "reproject" in cfg.processes.data_assimilation.optimization.obstacle_constraint:

            if "icemask" in cfg.processes.data_assimilation.cost_list:
                state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)

            if "thk" in cfg.processes.data_assimilation.control_list:
                state.thk = tf.where(state.thk < 0, 0, state.thk)

            if "slidingco" in cfg.processes.data_assimilation.control_list:
                state.slidingco = tf.where(state.slidingco < 0, 0, state.slidingco)

            if "arrhenius" in cfg.processes.data_assimilation.control_list:
                # Here we assume a minimum value of 1.0 for the arrhenius factor (should not be hard-coded)
                state.arrhenius = tf.where(state.arrhenius < 1.0, 1.0, state.arrhenius) 

        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, 
            method=cfg.processes.data_assimilation.divflux.method
        )

        # relaxation = 0.02
        # if relaxation>0:
        #     state.thk = tf.maximum(state.thk + relaxation * gaussian_filter_tf(state.divflux, sigma=2.0, kernel_size=13), 0)
        #     state.usurf = state.topg + state.thk

        #state.divflux = tf.where(ACT, state.divflux, 0.0)
