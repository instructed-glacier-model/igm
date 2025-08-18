import os
from typing import Union
import tensorflow as tf
from omegaconf import DictConfig
import importlib_resources
import igm.processes.iceflow.emulate.emulators as emulators
import logging


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

    dir_name = (
        "pinnbp"
        + "_"
        + str(cfg.processes.iceflow.numerics.Nz)
        + "_"
        + str(int(cfg.processes.iceflow.numerics.vert_spacing))
        + "_"
    )
    dir_name += (
        cfg.processes.iceflow.emulator.network.architecture
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_layers)
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_out_filter)
        + "_"
    )
    dir_name += (
        str(cfg.processes.iceflow.physics.dim_arrhenius) + "_" + str(int(1)) + "_" + L
    )

    return dir_name


def get_pretrained_emulator_path(cfg: DictConfig, state) -> str:

    cfg_emulator = cfg.processes.iceflow.emulator
    dir_name = get_emulator_path(cfg)

    dir_path = ""
    if cfg_emulator.name == "":
        print(importlib_resources.files(emulators).joinpath(dir_name))
        if os.path.exists(importlib_resources.files(emulators).joinpath(dir_name)):
            dir_path = importlib_resources.files(emulators).joinpath(dir_name)
            logging.info("Found pretrained emulator in the igm package: " + dir_name)
        else:
            raise ImportError(
                f"❌ No pretrained emulator found in the igm package with name <{dir_name}>."
            )
    else:
        dir_path = os.path.join(state.original_cwd, cfg_emulator.name)
        if os.path.exists(dir_path):
            logging.info(f"'-'*40 Found pretrained emulator: {cfg_emulator.name} ")
        else:
            raise ImportError(
                f"❌ No pretrained emulator found with path <{dir_path}>."
            )

    return dir_path


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
