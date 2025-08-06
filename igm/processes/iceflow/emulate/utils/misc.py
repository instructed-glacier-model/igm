import os
from typing import Union
import tensorflow as tf
from omegaconf import DictConfig

def get_effective_pressure_precentage(thk, percentage=0.8) -> tf.Tensor:
    p_i = 910  # kg/m^3, density of ice, # use IGM version not hardcoded
    g = 9.81  # m/s^2, gravitational acceleration # use IGM version not hardcoded

    ice_overburden_pressure = p_i * g * thk
    water_pressure = ice_overburden_pressure * percentage

    return ice_overburden_pressure - water_pressure


def get_emulator_path(cfg: DictConfig):
    L = (cfg.processes.iceflow.numerics.vert_basis == "Legendre") * "e" + (
        not cfg.processes.iceflow.numerics.vert_basis == "Legendre"
    ) * "a"

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
        str(cfg.processes.iceflow.physics.dim_arrhenius) + "_" + str(int(1)) + "_" + L
    )

    return direct_name

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
    fid.write(
        "%4.0f  %s \n"
        % (cfg.processes.iceflow.numerics.Nz, "# number of vertical grid point (Nz)")
    )
    fid.write(
        "%2.2f  %s \n"
        % (
            cfg.processes.iceflow.numerics.vert_spacing,
            "# param for vertical spacing (vert_spacing)",
        )
    )
    fid.close()
