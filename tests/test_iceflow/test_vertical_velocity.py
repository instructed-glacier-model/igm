import os
import tensorflow as tf
import pytest

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from omegaconf import OmegaConf


def test_vertical_velocity():

    state = State()
    cfg = load_yaml_recursive(
        os.path.join(igm.__path__[0], "conf"), exclude=["assimilations/pretraining"]
    )

    OmegaConf.update(cfg, "processes.iceflow.vertical_velocity.enabled", True)

    Nz, Ny, Nx = 10, 40, 30

    state.thk = tf.Variable(tf.ones((Ny, Nx)) * 200)
    state.topg = tf.Variable(tf.zeros((Ny, Nx)))
    state.usurf = state.thk + state.topg
    state.dX = tf.Variable(tf.ones((Ny, Nx)) * 100)
    state.dx = 100
    state.it = -1

    igm.processes.iceflow.initialize(cfg, state)
    igm.processes.iceflow.update(cfg, state)
    igm.processes.iceflow.finalize(cfg, state)

    assert hasattr(state, "W")
    assert tf.reduce_mean(state.W).numpy() < 10 * 10
