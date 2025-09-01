#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
import datetime
from tqdm import tqdm

def print_costs(cfg, state, cost, i):

    vol = ( np.sum(state.thk) * (state.dx**2) / 10**9 ).numpy()
    # mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    f = open('costs.dat','a')

    def bound(x):
        return min(x, 9999999)

    keys = list(cost.keys()) 
    if i == 0:
        L = [f"{key:>8}" for key in ["it","vol"]] + [f"{key:>12}" for key in keys]
        # print("Costs:     " + "   ".join(L))
        print("   ".join([f"{key:>12}" for key in keys]),file=f)

    # if i % cfg.processes.data_assimilation.output.freq == 0:
    #     L = [datetime.datetime.now().strftime("%H:%M:%S"),f"{i:0>{8}}",f"{vol:>8.4f}"] \
    #       + [f"{bound(cost[key].numpy()):>12.4f}" for key in keys]
    #     print("   ".join(L))

    print("   ".join([f"{bound(cost[key].numpy()):>12.4f}" for key in keys]),file=f)


def print_info_data_assimilation(cfg, state, cost, i):
    # Compute volume in Gt
    vol = (np.sum(state.thk) * (state.dx**2) / 1e9).numpy()

    # Initialize tqdm bar if needed
    if i % cfg.processes.data_assimilation.output.freq == 0:
        if hasattr(state, "pbar_costs"):
            state.pbar_costs.close()
        state.pbar_costs = tqdm(
            desc=" Data assim.", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}"
        )

    # Prepare the postfix dictionary
    if hasattr(state, "pbar_costs"):
        dic_postfix = {
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{i:04d}",
            "Vol": f"{vol:06.2f}",
        }
        for key in cost:
            value = cost[key].numpy()
            dic_postfix[key] = f"{min(value, 9999999):06.3f}"
        
        state.pbar_costs.set_postfix(dic_postfix)
        state.pbar_costs.update(1)

def save_rms_std(cfg, state):

    np.savetxt(
        "rms_std_vol.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
                state.vol,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        comments='',
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf       vol",
    )
