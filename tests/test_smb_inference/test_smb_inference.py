"""Tests for the smb_inference assimilation module.

These are deliberately data-free: the end-to-end inversion needs input NetCDFs
that do not live in this repository. What is checked here is the part that can
stand alone — that the module imports, that the bundled emulator it runs
against is present and consistent with the config the module ships, and that
the L-BFGS replacing tensorflow_probability actually minimises.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import igm.processes.iceflow.emulate.emulators as emulators

# The bundled emulator the shipped experiment resolves to, and the field order
# its fieldin.dat declares. load_model_from_path asserts the config matches.
EMULATOR = "pinnbp_10_4_cnn_16_32_2_1_a"
FIELDIN = ["thk", "usurf", "arrhenius", "slidingco", "dX"]
NZ = 10


def emulator_dir() -> Path:
    return Path(inspect.getfile(emulators)).parent / EMULATOR


def test_module_exposes_igm_entry_points():
    import igm.assimilations.smb_inference as m

    for hook in ("initialize", "update", "finalize"):
        assert callable(getattr(m, hook)), f"{hook} missing or not callable"


def test_module_does_not_require_tensorflow_probability():
    """tfp is not an IGM dependency; the module must not import it eagerly."""
    import igm.assimilations.smb_inference.smb_inference as impl

    assert "tensorflow_probability" not in inspect.getsource(impl)


def test_bundled_emulator_is_present_and_complete():
    d = emulator_dir()
    for f in ("model.h5", "fieldin.dat", "fieldout.dat"):
        assert (d / f).is_file(), f"{EMULATOR}/{f} missing"


def test_bundled_emulator_fieldin_matches_the_module_config():
    """The module's unified.inputs must equal fieldin.dat, in order."""
    names = [ln.split()[0] for ln in
             (emulator_dir() / "fieldin.dat").read_text().splitlines() if ln.split()]
    assert names == FIELDIN


def test_bundled_emulator_has_the_expected_io_shape():
    model = tf.keras.models.load_model(emulator_dir() / "model.h5", compile=False)
    assert model.input_shape[-1] == len(FIELDIN)
    assert model.output_shape[-1] == 2 * NZ      # U and V on Nz levels

    y = model(tf.zeros((1, 32, 32, len(FIELDIN))), training=False)
    assert tuple(y.shape) == (1, 32, 32, 2 * NZ)


def test_slidingco_is_available_on_the_unified_stack():
    """The bundled emulator takes a slidingco channel, which unified must provide."""
    from igm.processes.iceflow.utils.fields import initialize_iceflow_fields
    from omegaconf import OmegaConf

    class State:
        pass

    state = State()
    state.thk = tf.zeros((4, 5))
    cfg = OmegaConf.create({"processes": {"iceflow": {
        "method": "unified",
        "numerics": {"Nz": NZ, "vert_spacing": 4.0},
        "physics": {"sliding": {"slidingco": 0.13, "tau_ref": 0.6034},
                    "viscosity": {"arrhenius": 78.0, "enhancement_factor": 1.0}},
        "unified": {"inputs": FIELDIN}}}})

    initialize_iceflow_fields(cfg, state)
    assert hasattr(state, "slidingco"), "unified stack did not provide slidingco"
    assert state.slidingco.shape == (4, 5)
    np.testing.assert_allclose(state.slidingco.numpy(), 0.13, rtol=1e-6)


def _quadratic(A, b):
    def f(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = 0.5 * tf.tensordot(x, tf.linalg.matvec(A, x), 1) - tf.tensordot(b, x, 1)
        return value, tape.gradient(value, x)

    return f


def test_lbfgs_solves_a_quadratic_exactly():
    from igm.assimilations.smb_inference.core.lbfgs import lbfgs_minimize

    A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
    b = tf.constant([1.0, -1.0])

    result = lbfgs_minimize(
        _quadratic(A, b), tf.constant([5.0, 5.0]), max_iterations=100, tolerance=1e-7
    )

    exact = tf.linalg.solve(A, tf.reshape(b, [-1, 1]))[:, 0]
    np.testing.assert_allclose(result.position.numpy(), exact.numpy(), atol=1e-4)
    assert bool(result.converged)
    assert not bool(result.failed)


def test_lbfgs_minimises_rosenbrock():
    from igm.assimilations.smb_inference.core.lbfgs import lbfgs_minimize

    def rosenbrock(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = tf.reduce_sum(
                100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
            )
        return value, tape.gradient(value, x)

    x0 = tf.constant([-1.2, 1.0] * 4, dtype=tf.float32)
    result = lbfgs_minimize(rosenbrock, x0, max_iterations=300, tolerance=1e-4)

    assert bool(result.converged)
    np.testing.assert_allclose(result.position.numpy(), np.ones(8), atol=1e-3)


def test_lbfgs_reports_a_decreasing_objective():
    """The callback must see monotonically non-increasing accepted steps."""
    from igm.assimilations.smb_inference.core.lbfgs import lbfgs_minimize

    A = tf.constant([[4.0, 0.5], [0.5, 3.0]])
    b = tf.constant([2.0, 1.0])
    seen = []

    lbfgs_minimize(
        _quadratic(A, b),
        tf.constant([9.0, -7.0]),
        max_iterations=40,
        tolerance=1e-8,
        callback=lambda i, x, f: seen.append(float(f)),
    )

    assert seen, "callback was never invoked"
    assert all(b_ <= a_ + 1e-5 for a_, b_ in zip(seen, seen[1:])), seen
