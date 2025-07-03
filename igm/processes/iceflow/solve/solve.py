import numpy as np 
import tensorflow as tf 
from igm.utils.math.getmag3d import getmag3d 
from igm.processes.iceflow.energy_iceflow.energy_iceflow import iceflow_energy
from igm.processes.iceflow.utils import EarlyStopping, update_2d_iceflow_variables, print_info
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

    fieldin = [tf.expand_dims(vars(state)[f], axis=0) for f in cfg.processes.iceflow.emulator.fieldin]

    early_stopping = EarlyStopping(relative_min_delta=0.0002, patience=10)

    if cfg.processes.iceflow.solver.plot_sol:
        plt.ion()  # enable interactive mode
        state.fig = plt.figure(dpi=200)
        state.ax = state.fig.add_subplot(1, 1, 1)
        state.ax.axis("off")
        state.ax.set_aspect("equal")

    for i in range(cfg.processes.iceflow.solver.nbitmax):
        with tf.GradientTape() as t:
            t.watch(U)
            t.watch(V)

            C_shear, C_slid, C_grav, C_float = iceflow_energy(
                cfg, tf.expand_dims(U, axis=0), tf.expand_dims(V, axis=0), fieldin
            )

            C_shear_cost = tf.reduce_mean(C_shear)
            C_slid_cost  = tf.reduce_mean(C_slid)
            C_grav_cost  = tf.reduce_mean(C_grav)
            C_float_cost = tf.reduce_mean(C_float)

            COST = C_shear_cost + C_slid_cost + C_grav_cost  + C_float_cost

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
 
        state.optimizer.apply_gradients(zip(grads, [U, V]))

        velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)

        if state.it <= 1:    
            print_info(state, i, C_shear_cost.numpy(), C_slid_cost.numpy(), \
                                 C_grav_cost.numpy(), COST.numpy(), 
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

        fieldin = [
            tf.expand_dims(vars(state)[f], axis=0) for f in cfg.processes.iceflow.emulator.fieldin
        ]

        C_shear, C_slid, C_grav, C_float = iceflow_energy(
            cfg, tf.expand_dims(U, axis=0), tf.expand_dims(V, axis=0), fieldin
        )

        COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
             + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
            
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
        state.U, state.V, Cost_Glen = solve_iceflow_lbfgs(cfg, state, state.U, state.V)
    else:
        state.U, state.V, Cost_Glen = solve_iceflow(cfg, state, state.U, state.V)

    
    if cfg.processes.iceflow.force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.U / velbar_mag),
                state.U,
            ) 
        state.V = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.V / velbar_mag),
                state.V,
            ) 
        
    if len(cfg.processes.iceflow.solver.save_cost)>0:
        np.savetxt(cfg.processes.iceflow.emulator.output_directory+cfg.processes.iceflow.solver.save_cost+'-'+str(state.it)+'.dat', np.array(Cost_Glen),  fmt="%5.10f")

    state.COST_Glen = Cost_Glen[-1].numpy()

    update_2d_iceflow_variables(cfg, state)
 
