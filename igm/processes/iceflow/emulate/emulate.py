#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
import os

from igm.processes.iceflow.utils import fieldin_to_X, Y_to_UV, compute_PAD, print_info
from igm.processes.iceflow.utils import get_velbase, get_velsurf, get_velbar, force_max_velbar
from igm.processes.iceflow.energy.energy import iceflow_energy_XY
from igm.processes.iceflow.energy.sliding_laws.sliding_law import sliding_law_XY
from igm.processes.iceflow.emulate.neural_network import *
from igm.processes.iceflow.emulate import emulators
from igm.utils.math.getmag import getmag
import importlib_resources 
import igm  
import matplotlib.pyplot as plt
import matplotlib

def initialize_iceflow_emulator(cfg, state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.opti_retrain = getattr(tf.keras.optimizers,cfg.processes.iceflow.emulator.optimizer)(
            learning_rate=cfg.processes.iceflow.emulator.lr,
            epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
            clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm
        )
    else:
        state.opti_retrain = getattr(tf.keras.optimizers.legacy,cfg.processes.iceflow.emulator.optimizer)( 
            learning_rate=cfg.processes.iceflow.emulator.lr,
            epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
            clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm
        )

    direct_name = (
        "pinnbp"
        + "_"
        + str(cfg.processes.iceflow.numerics.Nz)
        + "_"
        + str(int(cfg.processes.iceflow.numerics.vert_spacing))
        + "_"
    )
    direct_name += (
        cfg.processes.iceflow.emulator.network.architecture
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_layers)
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_out_filter)
        + "_"
    )
    direct_name += (
        str(cfg.processes.iceflow.physics.dim_arrhenius)
        + "_"
        + str(int(1))
    )

    if cfg.processes.iceflow.emulator.pretrained:
        if cfg.processes.iceflow.emulator.name == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print(
                    "Found pretrained emulator in the igm package: " + direct_name
                )
            else:
                print("No pretrained emulator found in the igm package")
        else:
            dirpath = os.path.join(state.original_cwd, cfg.processes.iceflow.emulator.name)
            if os.path.exists(dirpath):
                print("----------------------------------> Found pretrained emulator: " + cfg.processes.iceflow.emulator.name)
            else:
                print("----------------------------------> No pretrained emulator found ")

        fieldin = []
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg.processes.iceflow.emulator.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile() 
    else:
        print("----------------------------------> No pretrained emulator, start from scratch.") 
        nb_inputs = len(cfg.processes.iceflow.emulator.fieldin) + (cfg.processes.iceflow.physics.dim_arrhenius == 3) * (
            cfg.processes.iceflow.numerics.Nz - 1
        )
        nb_outputs = 2 * cfg.processes.iceflow.numerics.Nz
        state.iceflow_model = getattr(igm.processes.iceflow.emulate.emulate, cfg.processes.iceflow.emulator.network.architecture)(
            cfg, nb_inputs, nb_outputs
        )

    print(state.iceflow_model.summary())

    # direct_name = 'pinnbp_10_4_cnn_16_32_2_1'        
    # dirpath = importlib_resources.files(emulators).joinpath(direct_name)
    # iceflow_model_pretrained = tf.keras.models.load_model(
    #     os.path.join(dirpath, "model.h5"), compile=False
    # )
    # N=16
    # pretrained_weights = [layer.get_weights() for layer in iceflow_model_pretrained.layers[:N]]
    # for i in range(N):
    #     state.iceflow_model.layers[i].set_weights(pretrained_weights[i])

def update_iceflow_emulated(cfg, state):
    # Define the input of the NN, include scaling

    Ny, Nx = state.thk.shape
    N = cfg.processes.iceflow.numerics.Nz

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]

    X = fieldin_to_X(cfg, fieldin)

    if cfg.processes.iceflow.emulator.exclude_borders>0:
        iz = cfg.processes.iceflow.emulator.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        
    if cfg.processes.iceflow.emulator.network.multiple_window_size==0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if cfg.processes.iceflow.emulator.exclude_borders>0:
        iz = cfg.processes.iceflow.emulator.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(cfg, Y)
    U = U[0]
    V = V[0]

    state.U = tf.where(state.thk > 0, U, 0)
    state.V = tf.where(state.thk > 0, V, 0)

    # If requested, the speeds are artifically upper-bounded
    if cfg.processes.iceflow.force_max_velbar > 0:
        force_max_velbar(cfg, state)

    state.uvelbase, state.vvelbase = get_velbase(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis)
    state.uvelsurf, state.vvelsurf = get_velsurf(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis)
    state.ubar, state.vbar = get_velbar(state.U, state.V, state.vert_weight, cfg.processes.iceflow.numerics.vert_basis)


def update_iceflow_emulator(cfg, state, it, pertubate=False):
 
    run_it = False
    if cfg.processes.iceflow.emulator.retrain_freq > 0:
       run_it = (it % cfg.processes.iceflow.emulator.retrain_freq == 0)
 
    warm_up = int(it <= cfg.processes.iceflow.emulator.warm_up_it)

    if (warm_up | run_it):

        state.COST_EMULATOR = []
        state.GRAD_EMULATOR = []
     
        fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]

        vert_disc = [vars(state)[f] for f in ['zeta', 'dzeta', 'P', 'dPdz']]

        XX = fieldin_to_X(cfg, fieldin) 

        if pertubate:
            XX = pertubate_X(cfg, XX)  

        X = split_into_patches(XX, cfg.processes.iceflow.emulator.framesizemax,
                                   cfg.processes.iceflow.emulator.split_patch_method)
 
        Ny = X.shape[-3]
        Nx = X.shape[-2]
        
        PAD = compute_PAD(cfg,Nx,Ny)

        if warm_up:
            nbit = cfg.processes.iceflow.emulator.nbit_init
            lr = cfg.processes.iceflow.emulator.lr_init
        else:
            nbit = cfg.processes.iceflow.emulator.nbit
            lr = cfg.processes.iceflow.emulator.lr

        state.opti_retrain.lr = lr 

        iz = cfg.processes.iceflow.emulator.exclude_borders 

        if cfg.processes.iceflow.emulator.plot_sol:
            plt.ion()  # enable interactive mode
            state.fig = plt.figure(dpi=200)
            state.ax = state.fig.add_subplot(1, 1, 1)
            state.ax.axis("off")
            state.ax.set_aspect("equal")

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape(persistent=True) as t:

                    if cfg.processes.iceflow.emulator.lr_decay < 1:
                        state.opti_retrain.lr = lr * (cfg.processes.iceflow.emulator.lr_decay ** (i / 1000))

                    Y = state.iceflow_model(tf.pad(X[i, :, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
                    
                    energy_list = iceflow_energy_XY(cfg, X[i, :, iz:Ny-iz, iz:Nx-iz, :], \
                                                         Y[:,    iz:Ny-iz, iz:Nx-iz, :], vert_disc)
                    
                    if len(cfg.processes.iceflow.physics.sliding_law) > 0:
                        basis_vectors, sliding_shear_stress = \
                            sliding_law_XY(cfg, X[i, :, iz:Ny-iz, iz:Nx-iz, :], \
                                                Y[:,    iz:Ny-iz, iz:Nx-iz, :] )
 
                    energy_mean_list = [tf.reduce_mean(en) for en in energy_list]

                    COST = tf.add_n(energy_mean_list)

                    cost_emulator = cost_emulator + COST

                    U, V = Y_to_UV(cfg, Y) 
                    velsurf_mag = getmag(*get_velsurf(U[0],V[0], cfg.processes.iceflow.numerics.vert_basis))

                    if warm_up:
                        print_info(state,epoch, cfg, [e.numpy() for e in energy_mean_list], 
                                                        tf.reduce_max(velsurf_mag).numpy())

                    if (epoch + 1) % 100 == 0:
                         
                        if cfg.processes.iceflow.emulator.plot_sol:
                            im = state.ax.imshow(
                                np.where(state.thk > 0, velsurf_mag, np.nan),
                                origin="lower",
                                cmap="turbo",
                                norm=matplotlib.colors.LogNorm(vmin=1,vmax=300)
                            )
                            if not hasattr(state, "already_set_cbar"):
                                state.cbar = plt.colorbar(im, label='velocity')
                                state.already_set_cbar = True
                            state.fig.canvas.draw()  # re-drawing the figure
                            state.fig.canvas.flush_events()  # to flush the GUI events
                            state.ax.set_title("epoch : " + str(epoch), size=15)


                grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                if len(cfg.processes.iceflow.physics.sliding_law) > 0:
                    sliding_gradients = t.gradient( basis_vectors,
                                                    state.iceflow_model.trainable_variables,
                                                    output_gradients=sliding_shear_stress )
                    grads = [ grad + (sgrad / tf.cast(Nx * Ny, tf.float32)) \
                                for grad, sgrad in zip(grads, sliding_gradients) ]

                # if (epoch + 1) % 100 == 0:
                #     values = [tf.norm(g) for g in grads]
                #     normalized = values / tf.reduce_sum(values) 
                #     percentages = [100 * v.numpy() for v in normalized] 
                #     print("Percentages:", " | ".join(f"{p:.1f}%" for p in percentages[::2]))

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

                grad_emulator = tf.linalg.global_norm(grads)

                del t 
 
            state.COST_EMULATOR.append(cost_emulator)
            state.GRAD_EMULATOR.append(grad_emulator)

    
        if len(cfg.processes.iceflow.emulator.save_cost)>0:
            np.savetxt(cfg.processes.iceflow.emulator.save_cost+'-'+str(it)+'.dat',
                    np.array(list(zip(state.COST_EMULATOR,state.GRAD_EMULATOR))), fmt="%5.10f")

def split_into_patches(X, nbmax, split_patch_method):
    """
    This function splits the input tensor into patches of size nbmax x nbmax.
    The patches are then stacked together to form a new tensor.
    If stack along axis 0, the adata will be streammed in a sequential way
    If stack along axis 1, the adata will be streammed in a parallel way by baches
    """
    XX = []
    ny = X.shape[1]
    nx = X.shape[2]
    sy = ny // nbmax + 1
    sx = nx // nbmax + 1
    ly = int(ny / sy)
    lx = int(nx / sx)

    for i in range(sx):
        for j in range(sy):
#            if tf.reduce_max(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :]) > 0:
            XX.append(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    if split_patch_method == "sequential":
        XXX = tf.stack(XX, axis=0)
    elif split_patch_method == "parrallel":
        XXX = tf.expand_dims(tf.concat(XX, axis=0), axis=0)

    return XXX

def pertubate_X(cfg, X):

    XX = [X]

    for i,f in enumerate(cfg.processes.iceflow.emulator.fieldin):

        vec = [tf.ones_like(X[:,:,:,i])*(i==j) for j in range(X.shape[3])]
        vec = tf.stack(vec, axis=-1)
 
        if hasattr(cfg.processes, "data_assimilation"):
            if f in cfg.processes.data_assimilation.control_list:
                XX.append(X + X*vec*0.2)
                XX.append(X - X*vec*0.2)
        else:
            if f in ["thk","usurf"]: 
                XX.append(X + X*vec*0.2)
                XX.append(X - X*vec*0.2)
 
    return tf.concat(XX, axis=0)


def save_iceflow_model(cfg, state):
    directory = "iceflow-model"
    
    import shutil

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(cfg.processes.iceflow.physics.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(cfg.processes.iceflow.emulator.fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in cfg.processes.iceflow.emulator.fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (cfg.processes.iceflow.numerics.Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (cfg.processes.iceflow.numerics.vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()

 