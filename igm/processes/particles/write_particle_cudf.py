#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import shutil
import numpy as np
import tensorflow as tf


def initialize_write_particle_cudf(cfg, state):

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


def update_write_particle_cudf(cfg, state):

    try:
        import cudf
        import cupy as cp
    except ImportError:
        raise ImportError("The 'particles' module requires the 'cudf' and 'cupy. Please install Them.")

    if state.saveresult:

        filename = os.path.join(
            "trajectories",
            "traj-" + "{:08.2f}".format(state.t.numpy()).replace('.', '-'),
        )

        array = tf.transpose(
            tf.stack(
                [
                    state.particle["id"],
                    state.particle["x"] + state.x[0],
                    state.particle["y"] + state.y[0],
                    state.particle["z"],
                    state.particle["r"],
                    state.particle["t"],
                ],
                axis=0,
            )
        )
        array = tf.experimental.dlpack.to_dlpack(array)
        array = cp.from_dlpack(array)
        df = cudf.DataFrame(array)
        df.columns = [
            "Id",
            "x",
            "y",
            "z",
            "rh",
            "t"
        ]  # for some reason, my header shows '# Id' for the numpy version but 'Id' for GPU... fyi
        if cfg.processes.particles.output.format == "csv":
            df.to_csv(f"{filename}.csv", index=False)
        elif cfg.processes.particles.output.format == "feather":
            df.to_feather(f"{filename}")
        elif cfg.processes.particles.output.format == "parquet":
            df.to_parquet(f"{filename}")
        else:
            raise ValueError(
                "Output format not supported. Please use 'csv', 'feather' (CPU version but still fast), or 'parquet'."
            )
            
        # ft = os.path.join("trajectories", "time.dat")
        # with open(ft, "a") as f:
        #     print(state.t.numpy(), file=f)

        if cfg.processes.particles.add_topography:
            filename_topography = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())),
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
            
            array = tf.experimental.dlpack.to_dlpack(array)
            array = cp.from_dlpack(array)
            df_topo = cudf.DataFrame(array)
            df_topo.columns = ["x", "y", "z"]
            
            if cfg.processes.particles.output.format == "csv":
                df_topo.to_csv(f"{filename_topography}.csv", index=False)
            elif cfg.processes.particles.output.format == "feather":
                df_topo.to_feather(f"{filename_topography}")
            elif cfg.processes.particles.output.format == "parquet":
                df_topo.to_parquet(f"{filename_topography}")
            else:
                raise ValueError(
                    "Output format not supported. Please use 'csv', 'feather' (CPU version but still fast), or 'parquet'."
                )
            