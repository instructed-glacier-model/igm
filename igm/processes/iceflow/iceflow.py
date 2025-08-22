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
import warnings

from igm.processes.iceflow.emulate.emulated import (
    get_emulated_inputs,
    update_iceflow_emulated,
    get_emulated_inputs,
)
from igm.processes.iceflow.emulate.emulator import (
    update_iceflow_emulator,
    initialize_iceflow_emulator,
    get_emulator_inputs,
)
from igm.processes.iceflow.emulate.utils import save_iceflow_model

from igm.processes.iceflow.utils.misc import is_retrain
from igm.processes.iceflow.utils.misc import initialize_iceflow_fields
from igm.processes.iceflow.utils.data_preprocessing import (
    match_fieldin_dimensions,
    prepare_data,
)
from igm.processes.iceflow.utils.vertical_discretization import (
    define_vertical_weight,
    compute_levels,
    compute_zeta_dzeta,
)

from igm.processes.iceflow.solve.solve import (
    initialize_iceflow_solver,
    update_iceflow_solved,
)
from igm.processes.iceflow.diagnostic.diagnostic import (
    initialize_iceflow_diagnostic,
    update_iceflow_diagnostic,
)
from igm.processes.iceflow.unified.unified import (
    initialize_iceflow_unified,
    update_iceflow_unified,
)
from igm.processes.iceflow.emulate.utils.misc import (
    get_effective_pressure_precentage,
)
from igm.processes.iceflow.energy.utils import gauss_points_and_weights, legendre_basis


class Iceflow:
    pass


def initialize(cfg, state):

    # Make sure this function is only called once
    if hasattr(state, "was_initialize_iceflow_already_called"):
        return
    else:
        state.was_initialize_iceflow_already_called = True

    # Create ice flow object
    iceflow = Iceflow()
    state.iceflow = iceflow

    # Parameters aliases
    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    # deinfe the fields of the ice flow such a U, V, but also sliding coefficient, arrhenius, ectt
    initialize_iceflow_fields(cfg, state)

    # Temporary fix for the effective pressure
    if not hasattr(state, "effective_pressure"):
        warnings.warn(
            f"Effective pressure not provided for sliding law {cfg_physics.sliding.law}. Using 0% of ice overburden pressure as default."
        )

        state.effective_pressure = get_effective_pressure_precentage(
            state.thk, percentage=0.0
        )
        state.effective_pressure = tf.where(
            state.effective_pressure < 1e-3, 1e-3, state.effective_pressure
        )

    # Set vertical discretization
    state.vert_weight = define_vertical_weight(
        cfg_numerics.Nz, cfg_numerics.vert_spacing
    )
    vertical_basis = cfg_numerics.vert_basis.lower()

    if vertical_basis == "lagrange":
        state.levels = compute_levels(
            cfg_numerics.Nz,
            cfg_numerics.vert_spacing,
        )
        state.zeta, state.dzeta = compute_zeta_dzeta(state.levels)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    elif vertical_basis == "legendre":
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=cfg_numerics.Nz)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = legendre_basis(
            state.zeta, order=state.zeta.shape[0]
        )
    elif vertical_basis == "sia":
        if cfg_numerics.Nz != 2:
            raise ValueError("❌ SIA vertical basis only supports Nz=2.")
        state.zeta, state.dzeta = gauss_points_and_weights(ord_gauss=5)
        state.Leg_P, state.Leg_dPdz, state.Leg_I = None, None, None
    else:
        raise ValueError(f"❌ Unknown vertical basis: <{cfg_numerics.vert_basis}>.")

    vert_disc = [
        vars(state)[f] for f in ["zeta", "dzeta", "Leg_P", "Leg_dPdz"]
    ]  # Lets please not hard code this as it affects every function inside...
    vert_disc = (vert_disc[0], vert_disc[1], vert_disc[2], vert_disc[3])
    state.vert_disc = vert_disc

    # Set ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        # define the emulator, and the optimizer
        initialize_iceflow_emulator(cfg, state)
    elif iceflow_method == "solved":
        # define the solver, and the optimizer
        initialize_iceflow_solver(cfg, state)
    elif iceflow_method == "diagnostic":
        # define the second velocity field
        initialize_iceflow_diagnostic(cfg, state)
    elif iceflow_method == "unified":
        # define the velocity through a mapping
        initialize_iceflow_unified(cfg, state)
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    if not iceflow_method in ["solved", "unified"]:

        fieldin = [vars(state)[f] for f in cfg_emulator.fieldin]
        if cfg_physics.dim_arrhenius == 3:
            fieldin = match_fieldin_dimensions(fieldin)
        elif cfg_physics.dim_arrhenius == 2:
            fieldin = tf.stack(fieldin, axis=-1)

        # Initial warm up for the emulator
        nbit = cfg_emulator.nbit_init
        lr = cfg_emulator.lr_init
        state.opti_retrain.lr = lr

        X, padding, Ny, Nx, iz = prepare_data(
            cfg, fieldin, pertubate=cfg_emulator.pertubate
        )

        data = get_emulator_inputs(state, nbit, lr)
        # vertical_discr = state.iceflow.vertical_discr
        state.cost_emulator = update_iceflow_emulator(
            data, X, padding, Ny, Nx, iz, vert_disc, state.iceflow.emulator_params
        )

        data = get_emulated_inputs(state)
        updated_variable_dict = update_iceflow_emulated(
            data, fieldin, state.iceflow.emulated_params
        )

        for key, value in updated_variable_dict.items():
            setattr(state, key, value)

    assert (cfg_emulator.exclude_borders == 0) | (
        cfg_emulator.network.multiple_window_size == 0
    )
    # if (cfg.processes.iceflow.emulator.exclude_borders==0) and (cfg.processes.iceflow.emulator.network.multiple_window_size==0):
    # raise ValueError("The emulator must exclude borders or use multiple windows, otherwise it will not work properly.")


def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method in ["emulated", "diagnostic"]:

        cfg_emulator = cfg.processes.iceflow.emulator

        fieldin = [vars(state)[f] for f in cfg_emulator.fieldin]

        if cfg.processes.iceflow.physics.dim_arrhenius == 3:
            fieldin = match_fieldin_dimensions(fieldin)
        elif cfg.processes.iceflow.physics.dim_arrhenius == 2:
            fieldin = tf.stack(fieldin, axis=-1)

        vert_disc = [
            vars(state)[f] for f in ["zeta", "dzeta", "Leg_P", "Leg_dPdz"]
        ]  # Lets please not hard code this as it affects every function inside...
        vert_disc = (vert_disc[0], vert_disc[1], vert_disc[2], vert_disc[3])

        warm_up = int(state.it <= cfg_emulator.warm_up_it)
        if warm_up:
            nbit = cfg_emulator.nbit_init
            lr = cfg_emulator.lr_init
        else:
            nbit = cfg_emulator.nbit
            lr = cfg_emulator.lr
        state.opti_retrain.lr = lr

        if (cfg_emulator.retrain_freq > 0) & (
            state.it > 0
        ):  # lets try to combine logic into one function...
            do_retrain = is_retrain(state.it, cfg)
            if do_retrain:
                X, padding, Ny, Nx, iz = prepare_data(
                    cfg, fieldin, pertubate=cfg_emulator.pertubate
                )
                data = get_emulator_inputs(state, nbit, lr)
                state.cost_emulator = update_iceflow_emulator(
                    data,
                    X,
                    padding,
                    Ny,
                    Nx,
                    iz,
                    vert_disc,
                    state.iceflow.emulator_params,
                )

        data = get_emulated_inputs(state)
        updated_variable_dict = update_iceflow_emulated(
            data, fieldin, state.iceflow.emulated_params
        )
        for key, value in updated_variable_dict.items():
            setattr(state, key, value)

        if iceflow_method == "diagnostic":
            update_iceflow_diagnostic(cfg, state)

    elif iceflow_method == "solved":
        update_iceflow_solved(cfg, state)

    elif iceflow_method == "unified":
        update_iceflow_unified(cfg, state)

    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")


def finalize(cfg, state):

    if cfg.processes.iceflow.emulator.save_model:
        save_iceflow_model(cfg, state)
