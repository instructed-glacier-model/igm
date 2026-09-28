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


def _batch_mean_quadratic_energy(U, V, inputs):
    """Average independent per-sample energies, as the IGM energy cost does."""
    del inputs
    shifted = tf.roll(U, shift=1, axis=-1) + tf.roll(V, shift=1, axis=-2)
    per_sample = tf.reduce_sum(
        U * U + V * V + 0.5 * U * shifted + 0.25 * V * V * V * V,
        axis=[1, 2, 3],
    )
    return tf.reduce_mean(per_sample)


@pytest.mark.parametrize("probe_batch", [7, 64])
def test_batched_probing_matches_sequential(probe_batch):
    shape = (1, 2, 5, 7)
    rng = np.random.default_rng(3)
    mapping = MappingIdentity(
        [],
        tf.constant(rng.normal(size=shape), tf.float64),
        tf.constant(rng.normal(size=shape), tf.float64),
        precision="double",
    )
    inputs = tf.zeros((1, 5, 7, 1), tf.float64)
    damping = tf.constant(0.0, tf.float64)
    sequential = MOLHOBandedADOperator(_batch_mean_quadratic_energy, mapping, "double")
    batched = MOLHOBandedADOperator(
        _batch_mean_quadratic_energy, mapping, "double", probe_batch=probe_batch
    )
    sequential.prepare(inputs, damping)
    batched.prepare(inputs, damping)
    np.testing.assert_allclose(
        batched._center.numpy(), sequential._center.numpy(), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        batched._edges.numpy(), sequential._edges.numpy(), rtol=1e-10, atol=1e-12
    )

    # hvp_many rows are the individual HVPs
    exact = ADOperator(_batch_mean_quadratic_energy, mapping, "double")
    vectors = tf.constant(rng.normal(size=(5, 2 * np.prod(shape))), tf.float64)
    many = exact.hvp_many(inputs, vectors, damping)
    for k in range(5):
        np.testing.assert_allclose(
            many[k].numpy(),
            exact.hvp(inputs, vectors[k], damping).numpy(),
            rtol=1e-10,
            atol=1e-12,
        )


def test_batched_probing_rejects_incompatible_probe_mode():
    shape = (1, 2, 3, 3)
    mapping = MappingIdentity(
        [], tf.zeros(shape, tf.float64), tf.zeros(shape, tf.float64), precision="double"
    )
    with pytest.raises(ValueError, match="probe_mode='autodiff'"):
        MOLHOBandedADOperator(
            _batch_mean_quadratic_energy,
            mapping,
            "double",
            probe_mode="fd",
            probe_batch=2,
        )


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

    graph = (
        tf.function(invert_spd_4x4)
        .get_concrete_function(
            center,
            tf.constant(1e-14, tf.float64),
        )
        .graph
    )
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


@pytest.mark.parametrize("ny", [2, 3, 4, 5])
def test_compact_molho_periodic_few_rows(ny):
    """periodic_ns flowlines: offsets alias when fewer than 3 rows are active."""
    shape = (1, 2, ny, 7)
    rng = np.random.default_rng(21)
    mapping = MappingIdentity(
        [PeriodicNS()],
        tf.constant(rng.normal(size=shape), tf.float64),
        tf.constant(rng.normal(size=shape), tf.float64),
        precision="double",
    )
    inputs = tf.zeros((1, 1, 1, 1, 1), tf.float64)
    damping = tf.constant(1e-15, tf.float64)
    exact = ADOperator(_quadratic_energy, mapping, "double")
    compact = MOLHOBandedADOperator(_quadratic_energy, mapping, "double")
    compact.prepare(inputs, damping)
    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), tf.float64)
    reference = exact.hvp(inputs, vector, damping)
    actual = compact.hvp(inputs, vector, damping)
    relative = tf.norm(reference - actual) / tf.norm(reference)
    assert float(relative) < 1e-11


@pytest.mark.parametrize("ny", [1, 2, 3, 4])
def test_symmetric_band_extraction_periodic_few_rows(ny):
    """duplicated_endpoints=False (multigrid levels): true period-ny wrap."""
    nx = 7
    n_components = 2
    rng = np.random.default_rng(8)
    center = tf.constant(rng.normal(size=(n_components, n_components)), tf.float64)
    center = 0.5 * (center + tf.transpose(center))
    couplings = {
        offset: tf.constant(rng.normal(size=(n_components, n_components)), tf.float64)
        for offset in ((1, 0), (0, 1), (1, 1), (1, -1))
    }

    def move(value: tf.Tensor, dy: int, dx: int) -> tf.Tensor:
        """Sample the (y + dy, x + dx) neighbour: periodic y, zero-padded x."""
        if dy:
            value = tf.roll(value, shift=-dy, axis=-2)
        if dx > 0:
            value = tf.pad(
                value[..., dx:], [[0, 0]] * (value.shape.rank - 1) + [[0, dx]]
            )
        elif dx < 0:
            value = tf.pad(
                value[..., :dx], [[0, 0]] * (value.shape.rank - 1) + [[-dx, 0]]
            )
        return value

    def operator(components: tf.Tensor) -> tf.Tensor:
        """Exact 9-point operator: true period-ny wrap in y, open in x."""
        result = tf.einsum("oi,...iyx->...oyx", center, components)
        for (dy, dx), weight in couplings.items():
            result += tf.einsum("oi,...iyx->...oyx", weight, move(components, dy, dx))
            result += tf.einsum(
                "oi,...iyx->...oyx", tf.transpose(weight), move(components, -dy, -dx)
            )
        return result

    packed_center, edges = extract_symmetric_bands_batched(
        operator,
        1,
        n_components,
        ny,
        nx,
        tf.float64,
        periodic_y=True,
        periodic_x=False,
        duplicated_endpoints=False,
    )
    stencil = SymmetricBandedStencil(
        packed_center,
        edges,
        periodic_y=True,
        periodic_x=False,
        duplicated_endpoints=False,
    )
    vector = tf.constant(rng.normal(size=(1, n_components, ny, nx)), tf.float64)
    reference = operator(vector)
    actual = stencil.apply(vector)
    relative = tf.norm(reference - actual) / tf.norm(reference)
    assert float(relative) < 1e-11


def test_alias_multipliers_are_identity_for_three_active_cells():
    """The aliasing weights must not touch grids with >= 3 active cells."""
    from igm.processes.iceflow.unified.operators.banded import (
        component_band_multipliers,
    )
    from igm.processes.iceflow.unified.operators.molho_banded import (
        symmetric_edge_multipliers,
    )

    cases = [
        dict(ny=4, nx=7, periodic_y=True, periodic_x=False),
        dict(ny=5, nx=5, periodic_y=True, periodic_x=True),
        dict(ny=2, nx=7, periodic_y=False, periodic_x=False),
        dict(ny=100, nx=200, periodic_y=True, periodic_x=False),
    ]
    for duplicated in (True, False):
        for case in cases:
            ny = case["ny"] + (1 if not duplicated and case["periodic_y"] else 0)
            args = {**case, "ny": ny, "duplicated_endpoints": duplicated}
            assert set(component_band_multipliers(**args)) == {1.0}, args
            assert set(symmetric_edge_multipliers(**args)) == {1.0}, args
