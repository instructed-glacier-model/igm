#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
 Quick notes about the code below:
 
 The goal of this module is to compute the ice flow velocity field
 using a deep-learning emulator of the Blatter-Pattyn model.
  
 The aim of this module is
   - to initialize the ice flow and its emulator in init_iceflow
   - to update the ice flow and its emulator in update_iceflow

In update_iceflow, we compute/update with function _update_iceflow_emulated,
and retraine the iceflow emaultor in function _update_iceflow_emulator

- in _update_iceflow_emulated, we baiscially gather together all input fields
of the emulator and stack all in a single tensor X, then we compute the output
with Y = iceflow_model(X), and finally we split Y into U and V

- in _update_iceflow_emulator, we retrain the emulator. For that purpose, we
iteratively (usually we do only one iteration) compute the output of the emulator,
compute the energy associated with the state of the emulator, and compute the
gradient of the energy with respect to the emulator parameters. Then we update
the emulator parameters with the gradient descent method (Adam optimizer).
Because this step may be memory consuming, we split the computation in several
patches of size cfg.processes.iceflow.emulator.framesizemax. This permits to
retrain the emulator on large size arrays.

Alternatively, one can solve the Blatter-Pattyn model using a solver using 
function _update_iceflow_solved. Doing so is not very different to retrain the
emulator as we minmize the same energy, however, with different controls,
namely directly the velocity field U and V instead of the emulator parameters.
"""

from igm.processes.iceflow.emulate.emulate import initialize_iceflow_emulator,update_iceflow_emulated, extract_state_for_emulated, UpdatedIceflowEmulatedParams
from igm.processes.iceflow.emulate.emulate import update_iceflow_emulator, save_iceflow_model, match_fieldin_dimensions
from igm.processes.iceflow.solve.solve import initialize_iceflow_solver, update_iceflow_solved
from igm.processes.iceflow.diagnostic.diagnostic import initialize_iceflow_diagnostic, update_iceflow_diagnostic
from igm.processes.iceflow.utils import initialize_iceflow_fields,compute_PAD
from igm.processes.iceflow.vert_disc import define_vertical_weight, compute_levels, compute_zeta_dzeta
from igm.processes.iceflow.energy.utils import gauss_points_and_weights, legendre_basis

import igm
    

def initialize(cfg, state):

    # This makes sure this function is only called once
    if hasattr(state, "was_initialize_iceflow_already_called"):
        return

    # deinfe the fields of the ice flow such a U, V, but also sliding coefficient, arrhenius, ectt
    initialize_iceflow_fields(cfg, state)

    if cfg.processes.iceflow.method == "emulated":
        # define the emulator, and the optimizer
        initialize_iceflow_emulator(cfg, state)
    elif cfg.processes.iceflow.method == "solved":
        # define the solver, and the optimizer
        initialize_iceflow_solver(cfg, state)    
    elif cfg.processes.iceflow.method == "diagnostic":
        # define the second velocity field
        initialize_iceflow_diagnostic(cfg,state)

    # create the vertica discretization
    state.vert_weight = define_vertical_weight(
        cfg.processes.iceflow.numerics.Nz,cfg.processes.iceflow.numerics.vert_spacing
                                              )
 
    if cfg.processes.iceflow.numerics.vert_basis == "Lagrange":
        state.levels = compute_levels(cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing)
        state.zeta, state.dzeta = compute_zeta_dzeta(state.levels)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=cfg.processes.iceflow.numerics.Nz)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = legendre_basis(state.zeta,order=state.zeta.shape[0]) 
    elif cfg.processes.iceflow.numerics.vert_basis == "SIA":
        assert cfg.processes.iceflow.numerics.Nz == 2 
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=5)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    else:
        raise ValueError(f"Unknown vertical basis: {cfg.processes.iceflow.numerics.vert_basis}")
    
    # padding is necessary when using U-net emulator
    state.PAD = compute_PAD(cfg, state.thk.shape[1],state.thk.shape[0])
    
    if not cfg.processes.iceflow.method == "solved":
        update_iceflow_emulator(cfg, state, 0)
        # update_iceflow_emulated(cfg, state)
        
        fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]
        if cfg.processes.iceflow.physics.dim_arrhenius == 3:
            fieldin = match_fieldin_dimensions(fieldin)
        elif cfg.processes.iceflow.physics.dim_arrhenius == 2:
            fieldin = tf.stack(fieldin, axis=-1)
        
        parameters = UpdatedIceflowEmulatedParams(
                Nz = cfg.processes.iceflow.numerics.Nz,
                arrhenius_dimension = cfg.processes.iceflow.physics.dim_arrhenius,
                fieldin = fieldin,
                exclude_borders = cfg.processes.iceflow.emulator.exclude_borders,
                multiple_window_size = cfg.processes.iceflow.emulator.network.multiple_window_size,
                force_max_velbar = cfg.processes.iceflow.force_max_velbar,
                vertical_basis  = cfg.processes.iceflow.numerics.vert_basis,
        )
            
        
        data = extract_state_for_emulated(state)
        updated_variable_dict = update_iceflow_emulated(data, fieldin, parameters)
        
        for key, value in updated_variable_dict.items():
            setattr(state, key, value)
            # if hasattr(state, key):
            # else:
                # state.logger.warning(f"Key {key} not found in state. Skipping update.")
         
    # Currently it is not supported to have the two working simulatanoutly
    assert (cfg.processes.iceflow.emulator.exclude_borders==0) | (cfg.processes.iceflow.emulator.network.multiple_window_size==0)
 
    # This makes sure this function is only called once
    state.was_initialize_iceflow_already_called = True

import tensorflow as tf

def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    if cfg.processes.iceflow.method in ["emulated","diagnostic"]:
        
        rng = igm.utils.profiling.srange("update_iceflow_emulator", "orange")
        if (cfg.processes.iceflow.emulator.retrain_freq > 0) & (state.it > 0):
            update_iceflow_emulator(cfg, state, state.it, 
                                    pertubate=cfg.processes.iceflow.emulator.pertubate)
        igm.utils.profiling.erange(rng)
        
        rng = igm.utils.profiling.srange("ANN Time step", "green")
        fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]
        
        if cfg.processes.iceflow.physics.dim_arrhenius == 3:
            fieldin = match_fieldin_dimensions(fieldin)
        elif cfg.processes.iceflow.physics.dim_arrhenius == 2:
            fieldin = tf.stack(fieldin, axis=-1)
        
        parameters = UpdatedIceflowEmulatedParams(
                Nz = cfg.processes.iceflow.numerics.Nz,
                arrhenius_dimension = cfg.processes.iceflow.physics.dim_arrhenius,
                fieldin = fieldin,
                exclude_borders = cfg.processes.iceflow.emulator.exclude_borders,
                multiple_window_size = cfg.processes.iceflow.emulator.network.multiple_window_size,
                force_max_velbar = cfg.processes.iceflow.force_max_velbar,
                vertical_basis  = cfg.processes.iceflow.numerics.vert_basis,
        )
        
        
        data = extract_state_for_emulated(state)
        # update_iceflow_emulated(data: Dict, fieldin: tf.Tensor, parameters)
        # extract_state_with_fieldin, UpdatedIceflowEmulatedParams
        updated_variable_dict = update_iceflow_emulated(data, fieldin, parameters)
        
        # print("Updated variable dict:", updated_variable_dict)
        
        for key, value in updated_variable_dict.items():
            setattr(state, key, value)
            # if hasattr(state, key):
            # else:
                # state.logger.warning(f"Key {key} not found in state. Skipping update.")
                
        igm.utils.profiling.erange(rng)

        if cfg.processes.iceflow.method == "diagnostic":
            update_iceflow_diagnostic(cfg, state)

    elif cfg.processes.iceflow.method == "solved":
        update_iceflow_solved(cfg, state)


def finalize(cfg, state):

    if cfg.processes.iceflow.emulator.save_model:
        save_iceflow_model(cfg, state)
   
 
  