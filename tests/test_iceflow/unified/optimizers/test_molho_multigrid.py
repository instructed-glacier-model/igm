import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.unified.bcs.dirichlet import DirichletBoundary
from igm.processes.iceflow.unified.bcs.frozen_bed import FrozenBed
from igm.processes.iceflow.unified.bcs.periodic_ns import PeriodicNS
from igm.processes.iceflow.unified.bcs.periodic_we import PeriodicWE
from igm.processes.iceflow.unified.mappings.identity import MappingIdentity
from igm.processes.iceflow.unified.operators.banded import periodic_axes
from igm.processes.iceflow.unified.preconditioners import (
    BarotropicMultigrid,
    BarotropicMultigridPreconditioner,
    GridTransfer,
    barotropic_mode,
    invert_spd_4x4,
)
from igm.processes.iceflow.unified.operators import (
    ADOperator,
    MOLHOBandedADOperator,
)
from igm.processes.iceflow.unified.operators.molho_banded import (
    SymmetricBandedStencil,
    extract_symmetric_bands_batched,
)
def _quadratic_energy(U, V, inputs):
    del inputs
    components = tf.concat([U, V], axis=1)
    point = 0.5 * tf.reduce_sum(components * components)
    point += 0.1 * tf.reduce_sum(tf.square(tf.reduce_sum(components, axis=1)))
    dx = components[..., 1:] - components[..., :-1]
    dy = components[..., 1:, :] - components[..., :-1, :]
    return point + 0.25 * (tf.reduce_sum(dx * dx) + tf.reduce_sum(dy * dy))


@pytest.mark.parametrize(
    "bcs",
    [
        [],
        [DirichletBoundary(left=0.0, top=0.0)],
        [FrozenBed(tf.constant([1.0, 0.0], tf.float64))],
        [PeriodicNS(), PeriodicWE()],
    ],
    ids=["none", "dirichlet", "frozen-bed", "periodic"],
)
@pytest.mark.parametrize("probe_mode", ["autodiff", "forward"])
def test_compact_molho_matches_exact_hvp(bcs, probe_mode):
    shape = (1, 2, 5, 7)
    rng = np.random.default_rng(14)
    mapping = MappingIdentity(
        bcs,
        tf.constant(rng.normal(size=shape), tf.float64),
        tf.constant(rng.normal(size=shape), tf.float64),
        precision="double",
    )
    inputs = tf.zeros((1, 1, 1, 1, 1), tf.float64)
    damping = tf.constant(1e-15, tf.float64)
    exact = ADOperator(_quadratic_energy, mapping, "double")
    compact = MOLHOBandedADOperator(
        _quadratic_energy,
        mapping,
        "double",
        probe_mode=probe_mode,
    )
    compact.prepare(inputs, damping)
    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), tf.float64)
    reference = exact.hvp(inputs, vector, damping)
    actual = compact.hvp(inputs, vector, damping)
    relative = tf.norm(reference - actual) / tf.norm(reference)
    assert float(relative) < 1e-11


def test_modified_ldl_matches_dense_inverse_without_eigh(monkeypatch):
    rng = np.random.default_rng(5)
    factors = rng.normal(size=(1, 6, 7, 4, 4))
    matrix = factors @ np.swapaxes(factors, -1, -2)
    matrix += 1e-8 * np.eye(4)
    center = tf.constant(np.transpose(matrix, [0, 3, 4, 1, 2]), tf.float64)

    monkeypatch.setattr(
        tf.linalg,
        "eigh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed-size LDL must not call eigh")
        ),
    )
    actual = invert_spd_4x4(center, tf.constant(1e-14, tf.float64))
    actual = np.transpose(actual.numpy(), [0, 3, 4, 1, 2])
    expected = np.linalg.inv(matrix)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-6)

    graph = tf.function(invert_spd_4x4).get_concrete_function(
        center,
        tf.constant(1e-14, tf.float64),
    ).graph
    assert all("Eig" not in operation.type for operation in graph.get_operations())


def _laplace_stencil(ny, nx, dtype=tf.float64):
    center = np.zeros((3, 1, ny, nx))
    center[0] = center[2] = 5.0
    edges = np.zeros((4, 1, 2, 2, ny, nx))
    edges[0, :, 0, 0] = edges[0, :, 1, 1] = -1.0
    edges[1, :, 0, 0] = edges[1, :, 1, 1] = -1.0
    return tf.constant(center, dtype), tf.constant(edges, dtype)


@pytest.mark.parametrize("periodic", [False, True])
def test_bilinear_galerkin_stencil_matches_transfer(periodic):
    ny, nx = 9, 11
    center, edges = _laplace_stencil(ny, nx)
    fine = SymmetricBandedStencil(
        center,
        edges,
        periodic_y=periodic,
        periodic_x=periodic,
        duplicated_endpoints=False,
    )
    transfer = GridTransfer(
        ny,
        nx,
        periodic_y=periodic,
        periodic_x=periodic,
        coarse_size=4,
    )
    coarse_ny, coarse_nx = transfer.coarse_shape
    coarse_center, coarse_edges = extract_symmetric_bands_batched(
        lambda value: transfer.restrict(fine.apply_many(transfer.prolong(value))),
        1,
        2,
        coarse_ny,
        coarse_nx,
        tf.float64,
        periodic_y=periodic,
        periodic_x=periodic,
        duplicated_endpoints=False,
    )
    coarse = SymmetricBandedStencil(
        coarse_center,
        coarse_edges,
        periodic_y=periodic,
        periodic_x=periodic,
        duplicated_endpoints=False,
    )
    value = tf.random.stateless_normal(
        (1, 2, coarse_ny, coarse_nx), (3, 8), dtype=tf.float64
    )
    expected = transfer.restrict(fine.apply(transfer.prolong(value)))
    np.testing.assert_allclose(coarse.apply(value), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_barotropic_vcycle_is_symmetric_positive(periodic):
    ny, nx = 9, 11
    center, edges = _laplace_stencil(ny, nx)
    multigrid = BarotropicMultigrid(
        1,
        ny,
        nx,
        tf.float64,
        periodic_y=periodic,
        periodic_x=periodic,
        smoother_weight=2.0 / 3.0,
        smoother_steps=1,
        coarse_size=4,
    )
    multigrid.update(center, edges)
    left = tf.random.stateless_normal((1, 2, ny, nx), (2, 3), dtype=tf.float64)
    right = tf.random.stateless_normal((1, 2, ny, nx), (4, 5), dtype=tf.float64)
    pre_left = multigrid.apply(left)
    pre_right = multigrid.apply(right)
    np.testing.assert_allclose(
        tf.reduce_sum(left * pre_right),
        tf.reduce_sum(right * pre_left),
        rtol=1e-11,
        atol=1e-11,
    )
    assert float(tf.reduce_sum(left * pre_left)) > 0.0


def test_barotropic_mode_respects_frozen_bed():
    shape = (1, 2, 3, 3)
    zeros = tf.zeros(shape, tf.float64)
    free = MappingIdentity([], zeros, zeros, precision="double")
    frozen = MappingIdentity(
        [FrozenBed(tf.constant([1.0, 0.0], tf.float64))],
        zeros,
        zeros,
        precision="double",
    )
    np.testing.assert_allclose(
        barotropic_mode(free, tf.float64),
        [2.0**-0.5, 2.0**-0.5],
    )
    np.testing.assert_allclose(barotropic_mode(frozen, tf.float64), [0.0, 1.0])
    assert periodic_axes(free) == (False, False)


def test_periodic_full_smoother_has_stable_weight():
    shape = (1, 2, 5, 7)
    zeros = tf.zeros(shape, tf.float64)
    mapping = MappingIdentity(
        [PeriodicNS(), PeriodicWE()],
        zeros,
        zeros,
        precision="double",
    )
    preconditioner = BarotropicMultigridPreconditioner(
        mapping,
        shape,
        "double",
        smoother_weight=2.0 / 3.0,
        coarse_size=4,
    )
    assert float(preconditioner.smoother_weight) == pytest.approx(0.5)
    assert float(preconditioner.multigrid.smoother_weight) == pytest.approx(0.5)
