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
import tensorflow as tf

from igm.processes.iceflow.emulate.emulated import get_emulated_bag, update_iceflow_emulated
from igm.processes.iceflow.emulate.emulator import get_emulator_bag, update_iceflow_emulator, initialize_iceflow_emulator
from igm.processes.iceflow.emulate.utils import save_iceflow_model
 
from igm.processes.iceflow.utils.misc import initialize_iceflow_fields
from igm.processes.iceflow.utils.data_preprocessing import compute_PAD, match_fieldin_dimensions, prepare_X, get_fieldin
from igm.processes.iceflow.utils.vertical_discretization import define_vertical_weight, compute_levels, compute_zeta_dzeta

from igm.processes.iceflow.solve.solve import initialize_iceflow_solver, update_iceflow_solved
from igm.processes.iceflow.diagnostic.diagnostic import initialize_iceflow_diagnostic, update_iceflow_diagnostic
from igm.processes.iceflow.energy.utils import gauss_points_and_weights, legendre_basis
from igm.processes.iceflow.sliding import SlidingLaws, SlidingParams


class Iceflow:
    pass

def initialize(cfg, state):

    iceflow = Iceflow()
    state.iceflow = iceflow
    
    # This makes sure this function is only called once
    if hasattr(state, "was_initialize_iceflow_already_called"):
        return

    law = cfg.processes.iceflow.physics.sliding.law
    
    sliding_law_class = SlidingLaws[law]
    sliding_law_params_class = SlidingParams[law]
    sliding_law_params = sliding_law_params_class(
        staggered_grid=cfg.processes.iceflow.numerics.staggered_grid,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
        Nz=cfg.processes.iceflow.numerics.Nz,
        **cfg.processes.iceflow.physics.sliding[law]
    )

    sliding_law = sliding_law_class(sliding_law_params)
    state.iceflow.sliding_law = sliding_law

    # deinfe the fields of the ice flow such a U, V, but also sliding coefficient, arrhenius, ectt
    initialize_iceflow_fields(cfg, state)
    iceflow_method = cfg.processes.iceflow.method.lower()
    
    if iceflow_method == "emulated":
        # define the emulator, and the optimizer
        initialize_iceflow_emulator(cfg, state)
    elif iceflow_method == "solved":
        # define the solver, and the optimizer
        initialize_iceflow_solver(cfg, state)    
    elif iceflow_method == "diagnostic":
        # define the second velocity field
        initialize_iceflow_diagnostic(cfg,state)

    # create the vertica discretization
    state.vert_weight = define_vertical_weight(cfg.processes.iceflow.numerics.Nz,cfg.processes.iceflow.numerics.vert_spacing)
    vertical_basis = cfg.processes.iceflow.numerics.vert_basis.lower()

    if vertical_basis == "lagrange":
        state.levels = compute_levels(cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing)
        state.zeta, state.dzeta = compute_zeta_dzeta(state.levels)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    elif vertical_basis == "legendre":
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=cfg.processes.iceflow.numerics.Nz)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = legendre_basis(state.zeta,order=state.zeta.shape[0]) 
    elif vertical_basis == "sia":
        if cfg.processes.iceflow.numerics.Nz != 2:
            raise ValueError("SIA vertical basis only supports Nz=2.")
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=5)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    else:
        raise ValueError(f"Unknown vertical basis: {cfg.processes.iceflow.numerics.vert_basis}")
    
    # padding is necessary when using U-net emulator
    state.PAD = compute_PAD(cfg.processes.iceflow.emulator.network.multiple_window_size,
                            state.thk.shape[1],state.thk.shape[0])
        
    vert_disc = [vars(state)[f] for f in ["zeta", "dzeta", "Leg_P", "Leg_dPdz"]] # Lets please not hard code this as it affects every function inside...
    state.vert_disc = (vert_disc[0], vert_disc[1], vert_disc[2], vert_disc[3])
    
    if not cfg.processes.iceflow.method.lower() == "solved":
                
        fieldin = get_fieldin(cfg, state)
             
        warm_up = int(0 <= cfg.processes.iceflow.emulator.warm_up_it)
        nbit = cfg.processes.iceflow.emulator.nbit_init if warm_up else cfg.processes.iceflow.emulator.nbit
        lr = cfg.processes.iceflow.emulator.lr_init if warm_up else cfg.processes.iceflow.emulator.lr
 
        X = prepare_X(cfg, fieldin, pertubate=cfg.processes.iceflow.emulator.pertubate, split_into_patches=True)
        bag = get_emulator_bag(state, nbit, lr)
        state.cost_emulator = update_iceflow_emulator(bag, X, state.iceflow.emulator_params)
        
        X = prepare_X(cfg, fieldin, pertubate=False, split_into_patches=False)
        bag = get_emulated_bag(state)
        updated_variable_dict = update_iceflow_emulated(bag, X, state.iceflow.emulated_params)
        
        for key, value in updated_variable_dict.items():
            setattr(state, key, value)
         
    assert (cfg.processes.iceflow.emulator.exclude_borders==0) | (cfg.processes.iceflow.emulator.network.multiple_window_size==0)
    # if (cfg.processes.iceflow.emulator.exclude_borders==0) and (cfg.processes.iceflow.emulator.network.multiple_window_size==0):
        # raise ValueError("The emulator must exclude borders or use multiple windows, otherwise it will not work properly.")

    # This makes sure this function is only called once
    state.was_initialize_iceflow_already_called = True

def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    if cfg.processes.iceflow.method.lower() in ["emulated","diagnostic"]:

        fieldin = get_fieldin(cfg, state)

        warm_up = int(state.it <= cfg.processes.iceflow.emulator.warm_up_it)
        nbit = cfg.processes.iceflow.emulator.nbit_init if warm_up else cfg.processes.iceflow.emulator.nbit
        lr = cfg.processes.iceflow.emulator.lr_init if warm_up else cfg.processes.iceflow.emulator.lr
        
        if (cfg.processes.iceflow.emulator.retrain_freq > 0) & (state.it > 0): # lets try to combine logic into one function...
            run_it = (state.it % cfg.processes.iceflow.emulator.retrain_freq == 0)
            if run_it or warm_up:
                X = prepare_X(cfg, fieldin, pertubate=cfg.processes.iceflow.emulator.pertubate, split_into_patches=True)
                bag = get_emulator_bag(state, nbit, lr)
                state.cost_emulator = update_iceflow_emulator(bag, X, state.iceflow.emulator_params)

        X = prepare_X(cfg, fieldin, pertubate=False, split_into_patches=False)
        bag = get_emulated_bag(state)
        updated_variable_dict = update_iceflow_emulated(bag, X, state.iceflow.emulated_params)

        for key, value in updated_variable_dict.items():
            setattr(state, key, value)

        if cfg.processes.iceflow.method.lower() == "diagnostic":
            update_iceflow_diagnostic(cfg, state)

    elif cfg.processes.iceflow.method.lower() == "solved":
        update_iceflow_solved(cfg, state)


def finalize(cfg, state):

    if cfg.processes.iceflow.emulator.save_model:
        save_iceflow_model(cfg, state)
   
 
  