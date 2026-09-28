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


def test_a_sliding_floating_slab_moves_along_its_base():
    """A uniform slab sliding at speed U moves parallel to its base, so
    W = U dz_b/dx at every level, with z_b the ice draft, not the seabed."""
    from types import SimpleNamespace

    import numpy as np

    from igm.processes.iceflow.utils.vertical_discretization import (
        define_vertical_weight,
    )
    from igm.processes.iceflow.vertical_velocity.vertical_velocity_v2 import (
        compute_vertical_velocity_kinematic_v2,
    )

    cfg = load_yaml_recursive(
        os.path.join(igm.__path__[0], "conf"), exclude=["assimilations/pretraining"]
    )
    numerics = cfg.processes.iceflow.numerics
    Nz, Ny, Nx, dx, speed, slope = numerics.Nz, 8, 20, 100.0, 50.0, -1.0e-3
    x = np.arange(Nx, dtype=np.float32) * dx
    lsurf = np.broadcast_to(-178.0 + slope * x, (Ny, Nx)).astype(np.float32)
    state = SimpleNamespace(
        thk=tf.fill((Ny, Nx), 200.0),
        topg=tf.fill((Ny, Nx), -1000.0),  # a flat, deep seabed
        lsurf=tf.constant(lsurf),
        U=tf.fill((Nz, Ny, Nx), speed),
        V=tf.zeros((Nz, Ny, Nx)),
        dx=dx,
        vert_weight=define_vertical_weight(Nz, numerics.vert_spacing),
    )
    W = compute_vertical_velocity_kinematic_v2(cfg, state).numpy()
    np.testing.assert_allclose(W[:, :, 1:-1], speed * slope, rtol=1e-4)
