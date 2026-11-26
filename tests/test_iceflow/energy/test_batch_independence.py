"""The ice-flow energy of a sample does not depend on the other samples of its
batch, for every horizontal basis."""

import os

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.discretization import HorizontalDiscrs, VerticalDiscrs
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components

NAMES = ("thk", "usurf", "arrhenius", "tau_ref", "dX")
B, NZ, NY, NX = 3, 4, 12, 14


def _cfg(basis):
    defaults = load_yaml_recursive(
        os.path.join(igm.__path__[0], "conf"), exclude=["assimilations/pretraining"]
    )
    return OmegaConf.merge(
        defaults,
        {
            "processes": {
                "iceflow": {
                    "numerics": {"precision": "double", "basis_horizontal": basis}
                }
            }
        },
    )


def _inputs(seed):
    rng = np.random.default_rng(seed)
    thk = 100.0 + 400.0 * rng.random((B, NY, NX))
    topg = 1000.0 + 300.0 * rng.random((B, NY, NX))
    X = np.stack(
        [
            thk,
            topg + thk,
            78.0 + 20.0 * rng.random((B, NY, NX)),
            0.1 + 0.2 * rng.random((B, NY, NX)),
            np.full((B, NY, NX), 100.0),
        ],
        axis=-1,
    )
    U = 50.0 * rng.random((B, NZ, NY, NX))
    V = 30.0 * rng.random((B, NZ, NY, NX))
    return X, U, V


def _energy_and_gradient(cfg, X, U, V):
    numerics = cfg.processes.iceflow.numerics
    discr_h = HorizontalDiscrs[numerics.basis_horizontal](cfg)
    discr_v = VerticalDiscrs[numerics.basis_vertical.lower()](cfg)
    components = get_energy_components(cfg)
    U = tf.constant(U)
    with tf.GradientTape() as tape:
        tape.watch(U)
        energy = iceflow_energy_UV(
            NAMES, tf.constant(X), U, tf.constant(V), discr_h, discr_v, components
        )
        per_sample = tf.reduce_sum(tf.reduce_mean(energy, axis=[2, 3]), axis=0)
        total = tf.reduce_sum(per_sample)
    return per_sample.numpy(), tape.gradient(total, U).numpy()


@pytest.mark.unit
@pytest.mark.parametrize("basis", ["central", "q1", "p1", "mac"])
def test_batched_energy_equals_per_sample_energy(basis):
    cfg = _cfg(basis)
    X, U, V = _inputs(seed=1)

    energy, gradient = _energy_and_gradient(cfg, X, U, V)
    for i in range(B):
        energy_i, gradient_i = _energy_and_gradient(
            cfg, X[i : i + 1], U[i : i + 1], V[i : i + 1]
        )
        np.testing.assert_allclose(energy[i], energy_i[0], rtol=1e-12)
        np.testing.assert_allclose(gradient[i], gradient_i[0], rtol=1e-10, atol=1e-14)

    X_thick = X.copy()
    X_thick[1, ..., 0] *= 1.5
    X_thick[1, ..., 1] += 0.5 * X[1, ..., 0]
    _, gradient_thick = _energy_and_gradient(cfg, X_thick, U, V)
    for i in (0, 2):
        np.testing.assert_allclose(
            gradient_thick[i], gradient[i], rtol=1e-10, atol=1e-14
        )
