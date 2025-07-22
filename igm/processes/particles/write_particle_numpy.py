#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import shutil
import numpy as np
import tensorflow as tf

def initialize_write_particle_numpy(cfg, state):

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    if cfg.processes.particles.output.add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update_write_particle_numpy(cfg, state):
    if state.saveresult:

        f = os.path.join(
            "trajectories",
            "traj-" + "{:08.2f}".format(state.t.numpy()).replace('.', '-') + ".csv",
        )

        array = tf.transpose(
            tf.stack(
                [
                    state.particle["id"].numpy(),
                    (state.particle["x"]+ state.x[0]).numpy().astype(np.float64),
                    (state.particle["y"]+ state.y[0]).numpy().astype(np.float64),
                    state.particle["z"],
                    state.particle["r"],
                    state.particle["t"]
                ],
                axis=0,
            )
        )
        np.savetxt(
            f, array, delimiter=",", fmt="%.2f", header="Id,x,y,z,rh,t"
        )

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if cfg.processes.particles.output.add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:08.2f}".format(state.t.numpy()).replace('.', '-') + ".csv",
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


