import numpy as np 
import tensorflow as tf 
from igm.utils.math.getmag import getmag
from igm.processes.iceflow.energy.energy import iceflow_energy
from igm.processes.iceflow.energy.sliding_laws.sliding_law import sliding_law
from igm.processes.iceflow.utils import EarlyStopping, print_info
from igm.processes.iceflow.utils import get_velbase, get_velsurf, get_velbar, clip_max_velbar
import matplotlib.pyplot as plt
import matplotlib

def initialize_iceflow_solver(cfg,state):

    if int(tf.__version__.split(".")[1]) <= 10:
        state.optimizer = getattr(tf.keras.optimizers,cfg.processes.iceflow.solver.optimizer)(
            learning_rate=cfg.processes.iceflow.solver.step_size
        )
    else:
        state.optimizer = getattr(tf.keras.optimizers.legacy,cfg.processes.iceflow.solver.optimizer)(
            learning_rate=cfg.processes.iceflow.solver.step_size
        )

def solve_iceflow(cfg, state, U, V):
    """
    solve_iceflow
    """

    U = tf.Variable(U)
    V = tf.Variable(V)

    Cost_Glen = []

    fieldin = [vars(state)[f][None,...] for f in cfg.processes.iceflow.emulator.fieldin]

    vert_disc = [vars(state)[f] for f in ['zeta', 'dzeta', 'Leg_P', 'Leg_dPdz']]

    early_stopping = EarlyStopping(relative_min_delta=0.0002, patience=10)

    if cfg.processes.iceflow.solver.plot_sol:
        plt.ion()  # enable interactive mode
        state.fig = plt.figure(dpi=200)
        state.ax = state.fig.add_subplot(1, 1, 1)
        state.ax.axis("off")
        state.ax.set_aspect("equal")

    for i in range(cfg.processes.iceflow.solver.nbitmax):
        with tf.GradientTape(persistent=True) as t:
            t.watch(U)
            t.watch(V)

            energy_list = iceflow_energy(
                cfg, U[None,:,:,:], V[None,:,:,:], fieldin, vert_disc
            ) 

            if len(cfg.processes.iceflow.physics.sliding_law) > 0:
                basis_vectors, sliding_shear_stress = sliding_law(cfg, U[None,:,:,:], V[None,:,:,:], fieldin)

            energy_mean_list = [tf.reduce_mean(en) for en in energy_list]

            COST = tf.add_n(energy_mean_list)

            Cost_Glen.append(COST)

            # if (i + 1) % 100 == 0:
            #    print("---------- > ", tf.reduce_mean(C_shear).numpy(), tf.reduce_mean(C_slid).numpy(), tf.reduce_mean(C_grav).numpy(), tf.reduce_mean(C_float).numpy())

#            state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_float = C_float[0] 

            # Stop if the cost no longer decreases
            # if cfg.processes.iceflow.solver.stop_if_no_decrease:
            #     if i > 1:
            #         if Cost_Glen[-1] >= Cost_Glen[-2]:
            #             break

        grads = t.gradient(COST, [U, V])

        if len(cfg.processes.iceflow.physics.sliding_law) > 0:
            sliding_gradients = t.gradient(basis_vectors, [U, V], output_gradients=sliding_shear_stress )
            grads = [ grad + (sgrad / tf.cast(U.shape[-2] * U.shape[-1], tf.float32)) \
                        for grad, sgrad in zip(grads, sliding_gradients) ]
 
        state.optimizer.apply_gradients(zip(grads, [U, V]))
        
        velsurf_mag = getmag(*get_velsurf(U,V, cfg.processes.iceflow.numerics.vert_basis))

        if state.it <= 1:    
            print_info(state, i, cfg, [e.numpy() for e in energy_mean_list], 
                                         tf.reduce_max(velsurf_mag).numpy())
 
        if (i + 1) % 100 == 0:

            # print("solve :", i, COST.numpy(), np.max(velsurf_mag)) 

            if cfg.processes.iceflow.solver.plot_sol:
                im = state.ax.imshow(
                    np.where(state.thk > 0, velsurf_mag, np.nan),
                    origin="lower",
                    cmap="turbo",
                    norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000)
                )
                if not hasattr(state, "already_set_cbar"):
                    state.cbar = plt.colorbar(im, label='velocity')
                    state.already_set_cbar = True
                state.fig.canvas.draw()  # re-drawing the figure
                state.fig.canvas.flush_events()  # to flush the GUI events
                state.ax.set_title("i : " + str(i), size=15)

        del t 

        if early_stopping.should_stop(COST.numpy()): 
#            print("Early stopping at iteration", i)
            break

    U = tf.where(state.thk > 0, U, 0)
    V = tf.where(state.thk > 0, V, 0)

    return U, V, Cost_Glen

def solve_iceflow_lbfgs(cfg, state, U, V):

    U = tf.Variable(U)
    V = tf.Variable(V)

    import tensorflow_probability as tfp

    Cost_Glen = []
 
    def COST(UV):

        U = UV[0]
        V = UV[1]

        fieldin = [vars(state)[f][None,...] for f in cfg.processes.iceflow.emulator.fieldin]

        energy_list = iceflow_energy(cfg, U[None,...], V[None,...], fieldin)
 
        energy_mean_list = [tf.reduce_mean(en) for en in energy_list]

        COST = tf.add_n(energy_mean_list)

            
        return COST

    def loss_and_gradients_function(UV):
        with tf.GradientTape() as tape:
            tape.watch(UV)
            loss = COST(UV)
            Cost_Glen.append(loss)
            gradients = tape.gradient(loss, UV)
        return loss, gradients
    
    UV = tf.stack([U, V], axis=0) 

    optimizer = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_gradients_function,
            initial_position=UV,
            max_iterations=cfg.processes.iceflow.solver.nbitmax,
            tolerance=1e-8)
    
    UV = optimizer.position

    U = UV[0]
    V = UV[1]
 
    return U, V, Cost_Glen

def update_iceflow_solved(cfg, state):

    if cfg.processes.iceflow.solver.lbfgs:
        raise ValueError("solve_iceflow_lbfgs formely implemented, not working yet, will be updated.")
        state.U, state.V, Cost_Glen = solve_iceflow_lbfgs(cfg, state, state.U, state.V)
    else:
        state.U, state.V, Cost_Glen = solve_iceflow(cfg, state, state.U, state.V)

    force_max_velbar = cfg.processes.iceflow.force_max_velbar
    vert_basis = cfg.processes.iceflow.numerics.vert_basis
    if cfg.processes.iceflow.force_max_velbar > 0:
        state.U, state.V = clip_max_velbar(state.U, state.V, force_max_velbar, vert_basis, state.vert_weight)
        
    if len(cfg.processes.iceflow.solver.save_cost)>0:
        np.savetxt(cfg.processes.iceflow.emulator.output_directory+cfg.processes.iceflow.solver.save_cost+'-'+str(state.it)+'.dat', np.array(Cost_Glen),  fmt="%5.10f")

    state.COST_Glen = Cost_Glen[-1].numpy()

    state.uvelbase, state.vvelbase = get_velbase(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis)
    state.uvelsurf, state.vvelsurf = get_velsurf(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis)
    state.ubar, state.vbar = get_velbar(state.U, state.V, state.vert_weight, cfg.processes.iceflow.numerics.vert_basis)