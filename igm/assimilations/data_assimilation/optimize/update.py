#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

# from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.utils.grad.compute_divflux import compute_divflux
from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf
from ..cost_terms.total_cost import total_cost

# from igm.processes.iceflow.emulate.emulate import update_iceflow_emulator, save_iceflow_model
# from igm.processes.iceflow.utils.misc import is_retrain, prepare_data, get_emulator_data

from ..iceflow_dispatch import iceflow_evaluate

from ..utils import compute_forward_divflux
from ..utils import compute_flow_direction_for_anisotropic_smoothing_vel
from ..utils import compute_flow_direction_for_anisotropic_smoothing_usurf


def optimize_update(cfg, state, cost, i):

    # Patch-wise (out-of-core) inversion for grids exceeding patch_size:
    # the gradient tape runs window by window instead of on the full grid,
    # so peak GPU memory is bounded by the window size. See update_patched.py.
    patch_size = cfg.assimilations.data_assimilation.optimization.patch_size
    if patch_size and patch_size > 0:
        ny, nx = state.thk.shape
        if ny > patch_size or nx > patch_size:
            from .update_patched import optimize_update_patched
            return optimize_update_patched(cfg, state, cost, i)

    sc = {}
    sc["thk"] = cfg.assimilations.data_assimilation.scaling.thk
    sc["usurf"] = cfg.assimilations.data_assimilation.scaling.usurf
    sc[state.da_friction] = cfg.assimilations.data_assimilation.scaling[state.da_friction]
    sc["arrhenius"] = cfg.assimilations.data_assimilation.scaling.arrhenius

    for f in cfg.assimilations.data_assimilation.control_list:
        if cfg.assimilations.data_assimilation.fitting.log_slidingco & (f == state.da_friction):
            setattr(state, f + "_sc", tf.Variable(tf.sqrt(getattr(state, f) / sc[f])))
        else:
            new_value = getattr(state, f) / sc[f]

        # Reuse the same tf.Variable across iterations so the (Keras 3) Adam
        # optimizer keeps recognizing it. Recreating it each step yields a new
        # variable identity and Keras 3's apply_gradients rejects unknown vars.
        key = f + "_sc"
        existing = getattr(state, key, None)
        if isinstance(existing, tf.Variable) and existing.shape == new_value.shape:
            existing.assign(new_value)
        else:
            setattr(state, key, tf.Variable(new_value))

    with tf.GradientTape() as t:

        if cfg.assimilations.data_assimilation.optimization.step_size_decay < 1:
            state.optimizer.lr = (
                cfg.assimilations.data_assimilation.optimization.step_size
                * (
                    cfg.assimilations.data_assimilation.optimization.step_size_decay
                    ** (i / 100)
                )
            )

        # is necessary to remember all operation to derive the gradients w.r.t. control variables
        for f in cfg.assimilations.data_assimilation.control_list:
            t.watch(getattr(state, f + "_sc"))

        for f in cfg.assimilations.data_assimilation.control_list:
            if cfg.assimilations.data_assimilation.fitting.log_slidingco & (
                f == state.da_friction
            ):
                setattr(state, f, (getattr(state, f + "_sc") ** 2) * sc[f])
            else:
                setattr(state, f, getattr(state, f + "_sc") * sc[f])

        iceflow_evaluate(cfg, state)

        if (
            not cfg.assimilations.data_assimilation.regularization.smooth_anisotropy_factor
            == 1
        ):
            if (
                cfg.assimilations.data_assimilation.regularization.smooth_anisotropy_var
                == "vel"
            ):
                compute_flow_direction_for_anisotropic_smoothing_vel(state)
            elif (
                cfg.assimilations.data_assimilation.regularization.smooth_anisotropy_var
                == "usurf"
            ):
                compute_flow_direction_for_anisotropic_smoothing_usurf(state)

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 1, figsize=(16,32))
            # plt.quiver(state.flowdirx[::2,::2], state.flowdiry[::2,::2])
            # axs.axis("equal")
            # plt.savefig("flow_directions.png", bbox_inches='tight', dpi=200)
            # plt.close()

        cost_total = total_cost(cfg, state, cost, i)

        var_to_opti = []
        for f in cfg.assimilations.data_assimilation.control_list:
            var_to_opti.append(getattr(state, f + "_sc"))

        # Compute gradient of COST w.r.t. X
        grads = tf.Variable(t.gradient(cost_total, var_to_opti))

        # this serve to restict the optimization of controls to the mask
        if cfg.assimilations.data_assimilation.optimization.sole_mask:
            for ii in range(grads.shape[0]):
                if not state.da_friction == cfg.assimilations.data_assimilation.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                else:
                    grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
        else:
            for ii in range(grads.shape[0]):
                if not state.da_friction == cfg.assimilations.data_assimilation.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

        # One step of descent -> this will update input variable X
        state.optimizer.apply_gradients(
            zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
        )

        ###################

        # get back optimized variables in the pool of state.variables
        for f in cfg.assimilations.data_assimilation.control_list:
            if cfg.assimilations.data_assimilation.fitting.log_slidingco & (
                f == state.da_friction
            ):
                setattr(state, f, (getattr(state, f + "_sc") ** 2) * sc[f])
            else:
                setattr(state, f, getattr(state, f + "_sc") * sc[f])

        # add reprojection step to force obstacle constraints
        if (
            "reproject"
            in cfg.assimilations.data_assimilation.optimization.obstacle_constraint
        ):

            if "icemask" in cfg.assimilations.data_assimilation.cost_list:
                state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)

            if "thk" in cfg.assimilations.data_assimilation.control_list:
                state.thk = tf.where(state.thk < 0, 0, state.thk)

            fric = state.da_friction
            if fric in cfg.assimilations.data_assimilation.control_list:
                setattr(state, fric, tf.where(getattr(state, fric) < 0, 0, getattr(state, fric)))

            if "arrhenius" in cfg.assimilations.data_assimilation.control_list:
                # Here we assume a minimum value of 1.0 for the arrhenius factor (should not be hard-coded)
                state.arrhenius = tf.where(state.arrhenius < 1.0, 1.0, state.arrhenius)

        # Diagnostic divflux (saved/plotted): ALWAYS the forward transport
        # operator — the only divergence whose smoothness predicts the
        # forward start (see compute_forward_divflux).
        state.divflux = compute_forward_divflux(cfg, state)
