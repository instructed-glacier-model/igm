#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

from igm.utils.math.getmag import getmag
from igm.processes.iceflow.utils.velocities import get_velsurf, get_velbar
from igm.processes.iceflow.solve.solve import solve_iceflow, initialize_iceflow_solver
from igm.processes.iceflow.emulate.emulator import initialize_iceflow_emulator

import time

def initialize_iceflow_diagnostic(cfg,state):

    initialize_iceflow_emulator(cfg,state)
    
    initialize_iceflow_solver(cfg,state)

    state.diagno = []

    state.velsurf_mag_exa = tf.zeros_like(state.thk)
    state.velsurf_mag_app = tf.zeros_like(state.thk)

    state.tlast_diagno = tf.Variable(-(10**10), dtype="float32", trainable=False)
 
#        time_solve = time.time()
        # time_solve -= time.time()
        # time_solve *= -1
     

def update_iceflow_diagnostic(cfg, state): 

    if (state.t - state.tlast_diagno) >= 10: # cfg.processes.iceflow.diagnostic.update_freq

        time_solve = time.time()
 
        U, V, Cost_Glen = solve_iceflow(cfg, state, state.U, state.V)
 
        state.velsurf_mag_app = getmag(*get_velsurf(state.U,state.V, cfg.processes.iceflow.numerics.vert_basis))
        state.velsurf_mag_exa = getmag(*get_velsurf(U,V, cfg.processes.iceflow.numerics.vert_basis))

        time_solve -= time.time()
        time_solve *= -1

        COST_Emulator = state.COST_EMULATOR[-1].numpy()
        COST_Glen     = Cost_Glen[-1].numpy() 

        nb_it_solve = len(Cost_Glen)
        nb_it_emul = len(state.COST_EMULATOR)

        l1, l2 = computemisfit(state, state.thk, state.U - U, state.V - V,cfg.processes.iceflow.numerics.vert_basis)

        vol = np.sum(state.thk) * (state.dx**2) / 10**9

        #######
        #training_strenght = 15 * (l1 / 10)**2
        #training_strenght = np.clip(training_strenght, 0.1, 10.0)        
        #cfg.processes.iceflow.emulator.retrain_freq = int(1/min(training_strenght,1))
        #cfg.processes.iceflow.emulator.nbit         = int(max(training_strenght,1))

        state.diagno.append([state.t.numpy(), l1, l2, COST_Glen, COST_Emulator, 
                             nb_it_solve, nb_it_emul, vol])
 
        np.savetxt("errors_diagno.txt", np.stack(state.diagno), delimiter=",", fmt="%10.3f",
                header="time,l1,l2,COST_Glen,COST_Emulator,nb_it_solve,nb_it_emul,vol",
                comments='')
            
        state.tlast_diagno.assign(state.t)


def computemisfit(state, thk, U, V, vert_basis):
    ubar, vbar = get_velbar(U, V,state.vert_weight, vert_basis)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    # MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

    nl1diff = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2diff = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1diff.numpy(), np.sqrt(nl2diff)

def finalize_iceflow_diagnostic(cfg, state):
    pass
