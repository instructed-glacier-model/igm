#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

from .utils import compute_rms_std_optimization, apply_relaxation
from .optimize.initialize import optimize_initialize
from .optimize.update import optimize_update
from .optimize.update_lbfgs import optimize_update_lbfgs
from .outputs.output_ncdf import update_ncdf_optimize, output_ncdf_optimize_final
from .outputs.prints import print_costs, save_rms_std, print_info_data_assimilation
from .outputs.plots import update_plot_inversion, plot_cost_functions
from .outputs.write_vtp import update_vtp

from igm.processes.iceflow import initialize as iceflow_initialize
from .iceflow_dispatch import iceflow_retrain

def initialize(cfg, state):

    iceflow_initialize(cfg, state) # initialize the iceflow model

    optimize_initialize(cfg, state)

    # update_iceflow_emulator(cfg, state, 0) # initialize the emulator

    # Early stopping state
    patience = getattr(cfg.processes.data_assimilation.optimization, "patience", 0)
    velsurf_history = []

    # iterate over the optimization process
    for i in range(cfg.processes.data_assimilation.optimization.nbitmax+1):

        cost = {}

        if cfg.processes.data_assimilation.optimization.method == "ADAM":
            optimize_update(cfg, state, cost, i)
        elif cfg.processes.data_assimilation.optimization.method == "L-BFGS":
            optimize_update_lbfgs(cfg, state, cost, i)
        else:
            raise ValueError(f"Unknown optim. method: {cfg.processes.data_assimilation.optimization.method}")

        if i == cfg.processes.data_assimilation.optimization.nbitmax:
            if cfg.processes.data_assimilation.optimization.nb_relaxation_steps > 0:
                apply_relaxation(cfg, state)

        compute_rms_std_optimization(state, i)

        # retraining the iceflow model (emulator or unified solver)
        if cfg.processes.data_assimilation.optimization.retrain_iceflow_model:
            state.it = i + 1
            cost["glen"] = iceflow_retrain(cfg, state)

        print_costs(cfg, state, cost, i)
        print_info_data_assimilation(cfg, state,  cost, i)

        if i % cfg.processes.data_assimilation.output.freq == 0:
            if cfg.processes.data_assimilation.output.plot2d:
                update_plot_inversion(cfg, state, i)
            if cfg.processes.data_assimilation.output.save_iterat_in_ncdf:
                update_ncdf_optimize(cfg, state, i)
                update_vtp(cfg, state, i)

        # Early stopping: compare rolling average of velsurf cost over
        # the last 'patience' iters vs the best rolling average seen so far.
        # Stop if the recent average is >10% worse than the best (0 = disabled).
        if patience > 0 and "velsurf" in cost:
            velsurf_val = float(cost["velsurf"].numpy() if hasattr(cost["velsurf"], 'numpy') else cost["velsurf"])
            velsurf_history.append(velsurf_val)
            if i >= cfg.processes.data_assimilation.optimization.nbitmin \
               and len(velsurf_history) >= 2 * patience:
                recent_avg = sum(velsurf_history[-patience:]) / patience
                best_avg = min(
                    sum(velsurf_history[j:j+patience]) / patience
                    for j in range(len(velsurf_history) - patience)
                )
                if recent_avg > best_avg * 1.1:
                    print(f"[data_assimilation] Early stopping at iter {i}: "
                          f"recent avg velsurf ({recent_avg:.2f}) > "
                          f"1.1 * best avg ({best_avg:.2f}) over window={patience}")
                    break

        state.topg = state.usurf - state.thk

    if not cfg.processes.data_assimilation.output.save_result_in_ncdf=="":
        output_ncdf_optimize_final(cfg, state)

    try:
        plot_cost_functions()
    except Exception as e:
        print(f"[data_assimilation] plot_cost_functions failed: {e}")

    save_rms_std(cfg, state) 
 

def update(cfg, state):
    pass

def finalize(cfg, state):
    pass