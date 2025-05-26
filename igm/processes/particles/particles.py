#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil 
import tensorflow as tf 

from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf

def initialize(cfg, state):

    if cfg.processes.particles.tracking_method == "3d":
        if "vert_flow" not in cfg.processes:
            raise ValueError(
                "The 'vert_flow' module is required to use the 3d tracking method in the 'particles' module."
            )


    state.tlast_seeding = cfg.processes.particles.tlast_seeding_init

    # initialize trajectories
    state.particle_x = tf.Variable([])
    state.particle_y = tf.Variable([])
    state.particle_z = tf.Variable([])
    state.particle_r = tf.Variable([])
    state.particle_w = tf.Variable([])  # this is to give a weight to the particle
    state.particle_t = tf.Variable([])
    state.particle_englt = tf.Variable([])  # this computes the englacial time
    state.particle_topg = tf.Variable([])
    state.particle_thk = tf.Variable([])

    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk), trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk), trainable=False)

    # build the gridseed, we don't want to seed all pixels!
    state.gridseed = np.zeros_like(state.thk) == 1
    # uniform seeding on the grid
    rr = int(1.0 / cfg.processes.particles.density_seeding)
    state.gridseed[::rr, ::rr] = True

    if cfg.processes.particles.write_trajectories:
        initialize_write_particle(cfg, state)

def update(cfg, state):

    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required to use the particles module")

    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:

        seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        state.particle_x = tf.Variable(
            tf.concat([state.particle_x, state.nparticle_x], axis=-1), trainable=False
        )
        state.particle_y = tf.Variable(
            tf.concat([state.particle_y, state.nparticle_y], axis=-1), trainable=False
        )
        state.particle_z = tf.Variable(
            tf.concat([state.particle_z, state.nparticle_z], axis=-1), trainable=False
        )
        state.particle_r = tf.Variable(
            tf.concat([state.particle_r, state.nparticle_r], axis=-1), trainable=False
        )
        state.particle_w = tf.Variable(
            tf.concat([state.particle_w, state.nparticle_w], axis=-1), trainable=False
        )
        state.particle_t = tf.Variable(
            tf.concat([state.particle_t, state.nparticle_t], axis=-1), trainable=False
        )
        state.particle_englt = tf.Variable(
            tf.concat([state.particle_englt, state.nparticle_englt], axis=-1),
            trainable=False,
        )
        state.particle_topg = tf.Variable(
            tf.concat([state.particle_topg, state.nparticle_topg], axis=-1),
            trainable=False,
        )
        state.particle_thk = tf.Variable(
            tf.concat([state.particle_thk, state.nparticle_thk], axis=-1),
            trainable=False,
        )

        state.tlast_seeding = state.t.numpy()

    if (state.particle_x.shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        u = interpolate_bilinear_tf(
            tf.expand_dims(state.U, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        v = interpolate_bilinear_tf(
            tf.expand_dims(state.V, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        thk = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_thk = thk

        topg = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_topg = topg

        smb = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        zeta = _rhs_to_zeta(cfg, state.particle_r)  # get the position in the column
        I0 = tf.cast(
            tf.math.floor(zeta * (cfg.processes.iceflow.numerics.Nz - 1)),
            dtype="int32",
        )
        I0 = tf.minimum(
            I0, cfg.processes.iceflow.numerics.Nz - 2
        )  # make sure to not reach the upper-most pt
        I1 = I0 + 1
        zeta0 = tf.cast(I0 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")
        zeta1 = tf.cast(I1 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")

        lamb = (zeta - zeta0) / (zeta1 - zeta0)

        ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
        ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))

        wei = tf.zeros_like(u)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

        if cfg.processes.particles.tracking_method == "simple":
            # adjust the relative height within the ice column with smb
            state.particle_r = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle_r * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = topg + thk * state.particle_r

        elif cfg.processes.particles.tracking_method == "3d":
            # uses the vertical velocity w computed in the vert_flow module

            w = interpolate_bilinear_tf(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = state.particle_z + state.dt * tf.reduce_sum(
                wei * w, axis=0
            )

            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - topg) / thk
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle_r = tf.where(
                thk == 0, tf.ones_like(state.particle_r), state.particle_r
            )

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(
            state.particle_x, 0, state.x[-1] - state.x[0]
        )
        state.particle_y = tf.clip_by_value(
            state.particle_y, 0, state.y[-1] - state.y[0]
        )

        indices = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(
            tf.where(state.particle_r == 1, state.particle_w, 0), dtype="float32"
        )

        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), indices, updates
        )

        # compute the englacial time
        state.particle_englt = state.particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        #    if int(state.t)%10==0:
        #        print("nb of part : ",state.xpos.shape)

    if cfg.processes.particles.write_trajectories:
        update_write_particle(cfg, state)

def finalize(cfg, state):
    pass


def _zeta_to_rhs(cfg, zeta):
    return (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
        1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
    )


def _rhs_to_zeta(cfg, rhs):
    if cfg.processes.iceflow.numerics.vert_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(
            1
            + 4
            * (cfg.processes.iceflow.numerics.vert_spacing - 1)
            * cfg.processes.iceflow.numerics.vert_spacing
            * rhs
        )
        zeta = (DET - 1) / (2 * (cfg.processes.iceflow.numerics.vert_spacing - 1))

    #           temp = cfg.processes.iceflow.numerics.Nz*(DET-1)/(2*(cfg.processes.iceflow.numerics.vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,cfg.processes.iceflow.numerics.Nz-1),dtype='int32')

    return zeta


def seeding_particles(cfg, state):
    """
    here we define (xpos,ypos) the horiz coordinate of tracked particles
    and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (a bit more), where there is
    significant ice, and in some points of a regular grid state.gridseed
    (density defined by density_seeding)

    """

    # ! THK and SMB modules are required. Insert in the init function of the particles module (actually, don't because the modules can be
    # ! initialized in any order, and the particles module is not guaranteed to be initialized after the thk and smb modules)
    # ! Instead, insert it HERE when needed (although it might call it multiple times and be less efficient...)

    if not hasattr(state, "thk"):
        raise ValueError("The thk module is required to use the particles module")
    if not hasattr(state, "smb"):
        raise ValueError(
            "A smb module is required to use the particles module. Please use the built-in smb module or create a custom one that overwrites the 'state.smb' value."
        )

    #        This will serve to remove imobile particles, but it is not active yet.

    #        indices = tf.expand_dims( tf.concat(
    #                       [tf.expand_dims((state.ypos - state.y[0]) / state.dx, axis=-1),
    #                        tf.expand_dims((state.xpos - state.x[0]) / state.dx, axis=-1)],
    #                       axis=-1 ), axis=0)

    #        thk = interpolate_bilinear_tf(
    #                    tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
    #                    indices,indexing="ij",      )[0, :, 0]

    #        J = (thk>1)

    # here we seed where i) thickness is higher than 1 m
    #                    ii) the seeding field of geology.nc is active
    #                    iii) on the gridseed (which permit to control the seeding density)
    #                    iv) on the accumulation area
    I = (
        (state.thk > 1) & state.gridseed & (state.smb > 0)
    )  # here you may redefine how you want to seed particles
    state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle_z = state.usurf[I]  # z position of the particle
    state.nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
    state.nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
    state.nparticle_t = (
        tf.ones_like(state.X[I]) * state.t
    )  # "date of birth" of the particle (useful to compute its age)
    state.nparticle_englt = tf.zeros_like(
        state.X[I]
    )  # time spent by the particle burried in the glacier
    state.nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]  # z position of the bedrock under the particle


def initialize_write_particle(cfg, state):

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    if cfg.processes.particles.add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update_write_particle(cfg, state):
    if state.saveresult:

        f = os.path.join(
            "trajectories",
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
        )

        ID = tf.cast(tf.range(state.particle_x.shape[0]), dtype="float32")
        array = tf.transpose(
            tf.stack(
                [
                    ID,
                    state.particle_x.numpy().astype(np.float64)
                    + state.x[0].numpy().astype(np.float64),
                    state.particle_y.numpy().astype(np.float64)
                    + state.y[0].numpy().astype(np.float64),
                    state.particle_z,
                    state.particle_r,
                    state.particle_t,
                    state.particle_englt,
                    state.particle_topg,
                    state.particle_thk,
                ],
                axis=0,
            )
        )
        np.savetxt(
            f, array, delimiter=",", fmt="%.2f", header="Id,x,y,z,rh,t,englt,topg,thk"
        )

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if cfg.processes.particles.add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [
                        state.X[state.X > 1],
                        state.Y[state.X > 1],
                        state.usurf[state.X > 1],
                    ]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


