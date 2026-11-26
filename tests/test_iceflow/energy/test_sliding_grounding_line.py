"""Grounding-line treatment of the basal friction (physics.sliding.use_subgrid_gl).

``nodal`` zeroes the friction at floating nodes inside the energy.
``subgrid`` multiplies the inputs that scale the friction by the grounded
fraction of the area around each node when the unified inputs are read from
the state (``get_inputs_from_state``), and the energy applies no mask.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

import igm
from igm.processes.enthalpy.dissipation.utils import compute_friction_heat
from igm.processes.iceflow.discretization.horizontal import HorizontalDiscrs
from igm.processes.iceflow.discretization.vertical import VerticalDiscrs
from igm.processes.iceflow.energy.components.sliding import (
    SlidingComponents,
    SlidingParams,
    get_sliding_params_args,
)
from igm.processes.iceflow.utils.inputs import get_inputs_from_state
from igm.processes.thk.masks import mask_gr, weights_gr

DTYPE = tf.float64
RHO_ICE = 910.0
RHO_WATER = 1000.0
RATIO = RHO_WATER / RHO_ICE
LAWS = ["weertman", "budd", "mohr_coulomb", "regu_coulomb"]
BASES = ["central", "q1", "p1", "mac"]
# Inputs weighted with use_subgrid_gl: the field the nodal mask acts on
WEIGHTED = {
    "weertman": {"tau_ref"},
    "budd": {"tau_ref"},
    "regu_coulomb": {"tau_ref"},
    "mohr_coulomb": {"effective_pressure"},
}
INPUTS = ["thk", "usurf", "arrhenius", "tau_ref", "dX", "water_level"]
DEFAULTS = OmegaConf.load(
    Path(igm.__file__).parent / "conf" / "processes" / "iceflow.yaml"
)


def _cfg(law, basis="q1", use_subgrid_gl=None, use_mask_gr=True, **sliding):
    sliding = dict(sliding, law=law, use_mask_gr=use_mask_gr)
    if use_subgrid_gl is not None:
        sliding["use_subgrid_gl"] = use_subgrid_gl
    if law == "mohr_coulomb" and use_subgrid_gl:
        sliding.setdefault("tauc_min", 0.0)
    inputs = list(INPUTS)
    if law != "weertman":
        inputs.append("effective_pressure")
    return OmegaConf.merge(
        {"processes": DEFAULTS},
        {
            "processes": {
                "iceflow": {
                    "numerics": {
                        "precision": "double",
                        "basis_horizontal": basis,
                        "basis_vertical": "lagrange",
                        "Nz": 2,
                    },
                    "physics": {
                        "ice_density": RHO_ICE,
                        "water_density": RHO_WATER,
                        "sliding": sliding,
                    },
                    "unified": {"inputs": inputs},
                }
            }
        },
    )


def _cost(cfg, fieldin, U, V):
    physics = cfg.processes.iceflow.physics
    numerics = cfg.processes.iceflow.numerics
    law = physics.sliding.law
    component = SlidingComponents[law](
        SlidingParams[law](**get_sliding_params_args(cfg))
    )
    discr_h = HorizontalDiscrs[numerics.basis_horizontal](cfg)
    discr_v = VerticalDiscrs[numerics.basis_vertical](cfg)

    @tf.function
    def cost(U, V):
        return component.cost(U, V, fieldin, discr_h, discr_v)

    return cost(U, V).numpy()


def _fields(thk, topg, tau_ref=None, effective_pressure=None, dx=1000.0):
    """Ice-flow fields with the surface the thk module would reconstruct."""
    thk = np.asarray(thk, np.float64)
    topg = np.asarray(topg, np.float64)
    water_level = np.zeros_like(thk)
    lsurf = np.maximum(topg, water_level - thk / RATIO)
    if tau_ref is None:
        tau_ref = np.full_like(thk, 0.1)
    if effective_pressure is None:
        effective_pressure = np.full_like(thk, 1.0)
    return {
        "thk": tf.constant(thk, DTYPE),
        "usurf": tf.constant(lsurf + thk, DTYPE),
        "topg": tf.constant(topg, DTYPE),
        "water_level": tf.constant(water_level, DTYPE),
        "arrhenius": tf.constant(np.full_like(thk, 78.0), DTYPE),
        "tau_ref": tf.constant(tau_ref, DTYPE),
        "effective_pressure": tf.constant(effective_pressure, DTYPE),
        "dX": tf.constant(np.full_like(thk, dx), DTYPE),
    }


def _random_case(seed, kind, shape=(2, 7, 9)):
    rng = np.random.default_rng(seed)
    batch, ny, nx = shape
    thk = 400.0 + 100.0 * rng.random(shape)
    if kind == "grounded":
        topg = 50.0 + 100.0 * rng.random(shape)
    elif kind == "floating":
        topg = -2000.0 + 100.0 * rng.random(shape)
    else:
        x = np.linspace(0.0, 1.0, nx)[None, None, :]
        topg = -100.0 - 500.0 * x + 30.0 * rng.normal(size=shape)
    fieldin = _fields(
        thk,
        topg,
        tau_ref=0.05 + 0.1 * rng.random(shape),
        effective_pressure=0.5 + rng.random(shape),
    )
    U = tf.constant(50.0 + 200.0 * rng.random((batch, 2, ny, nx)), DTYPE)
    V = tf.constant(-20.0 + 40.0 * rng.random((batch, 2, ny, nx)), DTYPE)
    return fieldin, U, V


def _subgrid_fields(fieldin, law):
    """``fieldin`` with the inputs of ``law`` weighted, as get_inputs_from_state."""
    weight = weights_gr(fieldin["thk"], fieldin["topg"], fieldin["water_level"], RATIO)
    return {
        name: field * weight if name in WEIGHTED[law] else field
        for name, field in fieldin.items()
    }


@pytest.mark.unit
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("law", LAWS)
def test_nodal_reproduces_previous_cost_bitwise(law, basis):
    """nodal (the default) masks the friction at the nodes, as before.

    The reference is the unmasked law applied to pre-masked inputs.
    Mohr-Coulomb derives its friction from N, so with tauc_min = 0 masking N
    masks tau_c.
    """
    fieldin, U, V = _random_case(11, "mixed")
    sliding = {}
    if law == "mohr_coulomb":
        sliding["tauc_min"] = 0.0
    grounded = mask_gr(
        fieldin["thk"],
        fieldin["usurf"] - fieldin["thk"],
        fieldin["water_level"],
        RATIO,
    )
    masked = dict(fieldin)
    if law == "mohr_coulomb":
        masked["effective_pressure"] = fieldin["effective_pressure"] * grounded
    else:
        masked["tau_ref"] = fieldin["tau_ref"] * grounded
    reference = _cost(_cfg(law, basis, use_mask_gr=False, **sliding), masked, U, V)

    default = _cost(_cfg(law, basis, **sliding), fieldin, U, V)
    nodal = _cost(_cfg(law, basis, False, **sliding), fieldin, U, V)

    np.testing.assert_array_equal(default, reference)
    np.testing.assert_array_equal(nodal, reference)
    assert np.all(np.isfinite(nodal))


@pytest.mark.unit
@pytest.mark.parametrize("basis", BASES)
def test_regu_coulomb_has_no_friction_where_its_coefficient_vanishes(basis):
    """Masked (or weighted out) friction used to give 0/0 or inf - inf: NaN.

    The cost tends to C |u_b|^p / p -> 0 as C -> 0, which the law now returns
    exactly; cells with friction are untouched.
    """
    fieldin, U, V = _random_case(3, "floating")
    nodal = _cost(_cfg("regu_coulomb", basis, False), fieldin, U, V)
    np.testing.assert_array_equal(nodal, np.zeros_like(nodal))

    fieldin, U, V = _random_case(3, "grounded")
    nodal = _cost(_cfg("regu_coulomb", basis, False), fieldin, U, V)
    unmasked = _cost(_cfg("regu_coulomb", basis, use_mask_gr=False), fieldin, U, V)
    np.testing.assert_array_equal(nodal, unmasked)
    assert np.all(nodal > 0.0)


def _planar_share(a, b, lo, hi):
    """Exact grounded fraction of [lo, hi] for phi = a + b x."""
    if b == 0.0:
        return 1.0 if a > 0.0 else 0.0
    x0 = -a / b
    if b > 0.0:
        grounded = hi - np.clip(x0, lo, hi)
    else:
        grounded = np.clip(x0, lo, hi) - lo
    return grounded / (hi - lo)


@pytest.mark.unit
@pytest.mark.parametrize("along", ["x", "y"])
def test_weights_gr_are_exact_when_phi_varies_along_one_direction(along):
    """The area around node i spans [i - 1/2, i + 1/2] (half at the edges)."""
    rng = np.random.default_rng(0)
    n, m = 9, 4
    for a, b in rng.normal(size=(50, 2)) * [3.0, 1.0]:
        x = np.arange(n, dtype=np.float64)
        # thk = 0 and topg = phi / ratio: phi exactly, tolerance aside
        topg = np.broadcast_to((a + b * x) / RATIO, (m, n)).copy()
        if along == "y":
            topg = topg.T.copy()
        topg = tf.constant(topg)
        weight = weights_gr(
            tf.zeros_like(topg), topg, tf.zeros_like(topg), RATIO
        ).numpy()
        expected = np.array(
            [
                _planar_share(a, b, max(i - 0.5, 0.0), min(i + 0.5, n - 1.0))
                for i in range(n)
            ]
        )
        if along == "y":
            weight = weight.T
        # flotation_function subtracts 64 float32 ulps of thk + |topg|, which
        # moves the grounding line by ~1e-5 cell here
        np.testing.assert_allclose(weight, np.broadcast_to(expected, (m, n)), atol=1e-4)


@pytest.mark.unit
def test_weights_gr_bounds_and_extremes():
    fieldin, _, _ = _random_case(5, "mixed", shape=(3, 12, 15))
    weight = weights_gr(
        fieldin["thk"], fieldin["topg"], fieldin["water_level"], RATIO
    ).numpy()
    assert weight.shape == (3, 12, 15)
    assert np.all((weight >= 0.0) & (weight <= 1.0))
    assert np.any((weight > 0.0) & (weight < 1.0))
    for kind, value in (("grounded", 1.0), ("floating", 0.0)):
        fieldin, _, _ = _random_case(5, kind)
        weight = weights_gr(
            fieldin["thk"], fieldin["topg"], fieldin["water_level"], RATIO
        ).numpy()
        np.testing.assert_array_equal(weight, value)


@pytest.mark.unit
def test_weights_gr_vary_continuously_as_the_ice_thins():
    """A flowline on a bed deepening by 100 m per cell, thinned 0.5 m a step."""
    nx = 8
    topg = -100.0 - 100.0 * np.arange(nx)
    previous = None
    for thk in np.linspace(600.0, 300.0, 601):
        fields = _fields(np.full((3, nx), thk), np.broadcast_to(topg, (3, nx)))
        weight = weights_gr(
            fields["thk"], fields["topg"], fields["water_level"], RATIO
        ).numpy()
        if previous is not None:
            assert np.all(weight <= previous + 1e-12)
            # 0.5 m of thinning moves the grounding line by 0.5 / 110 cell
            assert np.abs(weight - previous).max() < 2.0 * 0.5 / (100.0 * RATIO)
        previous = weight


def _state(fieldin):
    return SimpleNamespace(**{name: field[0] for name, field in fieldin.items()})


def _raw_inputs(state, inputs):
    """The inputs stacked from ``state`` without any weighting."""
    return tf.stack([getattr(state, name) for name in inputs], axis=-1)


@pytest.mark.unit
@pytest.mark.parametrize("law", LAWS)
def test_inputs_are_read_unchanged_without_subgrid(law):
    fieldin, _, _ = _random_case(6, "mixed")
    state = _state(fieldin)
    for cfg in (
        _cfg(law, use_subgrid_gl=False),
        _cfg(law, use_subgrid_gl=True, use_mask_gr=False),
    ):
        inputs = cfg.processes.iceflow.unified.inputs
        np.testing.assert_array_equal(
            get_inputs_from_state(cfg, state).numpy(),
            _raw_inputs(state, inputs).numpy(),
        )


@pytest.mark.unit
@pytest.mark.parametrize("law", LAWS)
def test_subgrid_weights_exactly_the_inputs_that_scale_the_friction(law):
    fieldin, _, _ = _random_case(6, "mixed")
    state = _state(fieldin)
    cfg = _cfg(law, use_subgrid_gl=True)
    inputs = list(cfg.processes.iceflow.unified.inputs)
    raw = _raw_inputs(state, inputs).numpy()
    weighted = get_inputs_from_state(cfg, state).numpy()
    weight = weights_gr(state.thk, state.topg, state.water_level, RATIO).numpy()
    for k, name in enumerate(inputs):
        if name in WEIGHTED[law]:
            np.testing.assert_array_equal(weighted[..., k], raw[..., k] * weight)
        else:
            np.testing.assert_array_equal(weighted[..., k], raw[..., k])
    assert get_sliding_params_args(cfg)["use_mask_gr"] is False


@pytest.mark.unit
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("law", LAWS)
def test_subgrid_matches_nodal_away_from_the_grounding_line(law, basis):
    """Fully grounded: weight 1, the nodal friction. Fully floating: none."""
    for kind in ("grounded", "floating"):
        fieldin, U, V = _random_case(5, kind)
        nodal = _cost(_cfg(law, basis, False, tauc_min=0.0), fieldin, U, V)
        subgrid = _cost(_cfg(law, basis, True), _subgrid_fields(fieldin, law), U, V)
        np.testing.assert_allclose(subgrid, nodal, rtol=1e-14, atol=0.0)
        if kind == "floating":
            np.testing.assert_array_equal(subgrid, np.zeros_like(subgrid))


@pytest.mark.unit
@pytest.mark.parametrize("law", LAWS)
def test_subgrid_friction_and_gradients_are_finite_across_the_grounding_line(law):
    fieldin, U, V = _random_case(8, "mixed")
    weighted = _subgrid_fields(fieldin, law)
    cfg = _cfg(law, "q1", True)
    component = SlidingComponents[law](
        SlidingParams[law](**get_sliding_params_args(cfg))
    )
    discr_h = HorizontalDiscrs["q1"](cfg)
    discr_v = VerticalDiscrs["lagrange"](cfg)
    with tf.GradientTape() as tape:
        tape.watch(U)
        cost = component.cost(U, V, weighted, discr_h, discr_v)
        total = tf.reduce_sum(cost)
    assert np.all(np.isfinite(cost.numpy()))
    assert np.all(np.isfinite(tape.gradient(total, U).numpy()))


def _heat_state(thk, topg, tau_ref):
    """The fields compute_friction_heat reads, on one 2-D grid (Nz = 2)."""
    fieldin = _fields(thk, topg, tau_ref=tau_ref)
    ny, nx = thk.shape
    discr_v = VerticalDiscrs["lagrange"](_cfg("weertman"))
    rng = np.random.default_rng(3)
    return SimpleNamespace(
        thk=fieldin["thk"],
        usurf=fieldin["usurf"],
        topg=fieldin["topg"],
        water_level=fieldin["water_level"],
        tau_ref=fieldin["tau_ref"],
        dX=fieldin["dX"],
        U=tf.constant(50.0 + 200.0 * rng.random((2, ny, nx)), DTYPE),
        V=tf.constant(-20.0 + 40.0 * rng.random((2, ny, nx)), DTYPE),
        iceflow=SimpleNamespace(discr_v=discr_v),
    )


@pytest.mark.unit
def test_friction_heat_follows_the_subgrid_weight():
    """nodal heat is unchanged (floating nodes are zeroed by the solver)."""
    ny, nx = 4, 6
    x = np.arange(nx, dtype=np.float64)[None, :] * np.ones((ny, 1))
    thk = np.full((ny, nx), 500.0)
    # Grounding line at x = 2.3, grounded to the west
    topg = -thk / RATIO + 200.0 * (2.3 - x)
    state = _heat_state(thk, topg, 0.05 + 0.02 * x)

    def heat(use_subgrid_gl, use_mask_gr=True):
        cfg = _cfg("weertman", "q1", use_subgrid_gl, use_mask_gr=use_mask_gr)
        return compute_friction_heat(cfg, state).numpy()

    unmasked = heat(False, use_mask_gr=False)
    np.testing.assert_array_equal(heat(False), unmasked)
    weight = weights_gr(state.thk, state.topg, state.water_level, RATIO).numpy()
    np.testing.assert_allclose(heat(True), unmasked * weight, rtol=1e-12)
    assert np.any((weight > 0.0) & (weight < 1.0))
