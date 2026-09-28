#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The bmb process: assembly of grounded and ocean melt, switches, dispatch."""

import numpy as np
import pytest
import tensorflow as tf

import igm.processes.bmb as bmb_package
from igm.common.runner.modules.src import _build_provider_map, check_module_needs
from igm.processes.bmb import bmb
from igm.processes.bmb.laws import available_melt_laws

pytestmark = [pytest.mark.fast, pytest.mark.unit]

GROUNDED_MELT = 0.02


def _run(cfg, state):
    bmb.initialize(cfg, state)
    bmb.update(cfg, state)
    return state


def _channel_state(cfg_factory, state_factory, channel_factory, **bmb_cfg):
    cfg = cfg_factory(**bmb_cfg)
    thk, topg = channel_factory(front=32)
    state = state_factory(cfg, thk, topg, basal_melt_rate=GROUNDED_MELT, ubar=300.0)
    return cfg, state


def test_grounded_and_ocean_melt_are_assembled_on_their_side(
    cfg_factory, state_factory, channel_factory
):
    cfg, state = _channel_state(
        cfg_factory,
        state_factory,
        channel_factory,
        prescribed={"form": "constant", "value": 5.0},
    )
    _run(cfg, state)
    b = state.bmb.numpy()
    grounded_fraction = state.grounded_fraction.numpy()

    interior_grounded = grounded_fraction == 1.0
    interior_shelf = (grounded_fraction == 0.0) & (state.thk.numpy() > 0)
    assert interior_grounded.any() and interior_shelf.any()
    np.testing.assert_allclose(b[interior_grounded], -GROUNDED_MELT)
    np.testing.assert_allclose(b[interior_shelf], -5.0)
    np.testing.assert_array_equal(b[:, 32:], 0.0)  # no ice, no mass balance
    np.testing.assert_allclose(state.shelf_melt_rate.numpy()[interior_shelf], 5.0)


def test_switches(cfg_factory, state_factory, channel_factory):
    kwargs = dict(prescribed={"form": "constant", "value": -2.0})
    cfg, state = _channel_state(cfg_factory, state_factory, channel_factory, **kwargs)
    shelf = _run(cfg, state).grounded_fraction.numpy() == 0.0
    shelf &= state.thk.numpy() > 0
    np.testing.assert_array_equal(state.bmb.numpy()[shelf], 0.0)  # clipped

    cfg.processes.bmb.allow_refreezing = True
    cfg.processes.bmb.melt_enhancer = 3.0
    np.testing.assert_allclose(_run(cfg, state).bmb.numpy()[shelf], 6.0)

    cfg.processes.bmb.include_grounded_melt = False
    grounded = _run(cfg, state).grounded_fraction.numpy() == 1.0
    np.testing.assert_array_equal(state.bmb.numpy()[grounded], 0.0)


@pytest.mark.parametrize("treatment", ["nmp", "fmp", "pmp"])
def test_a_tidewater_front_keeps_its_grounded_melt(
    treatment, cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory(
        prescribed={"form": "constant", "value": 5.0},
        grounding_line={"treatment": treatment},
    )
    thk, topg = channel_factory(front=32)
    thk[:, :32] = 1500.0  # grounded up to the front, then ice-free ocean
    state = state_factory(cfg, thk, topg, basal_melt_rate=GROUNDED_MELT)
    _run(cfg, state)
    np.testing.assert_array_equal(state.grounded_fraction.numpy()[:, :32], 1.0)
    np.testing.assert_allclose(state.bmb.numpy()[:, :32], -GROUNDED_MELT)


def test_the_law_is_reevaluated_every_update_freq_years(
    cfg_factory, state_factory, channel_factory
):
    cfg, state = _channel_state(
        cfg_factory,
        state_factory,
        channel_factory,
        update_freq=10.0,
        prescribed={"form": "constant", "value": 1.0},
    )
    _run(cfg, state)
    cfg.processes.bmb.prescribed.value = 4.0
    state.t.assign(5.0)
    bmb.update(cfg, state)
    assert state.shelf_melt_rate.numpy().max() == 1.0
    state.t.assign(10.0)
    bmb.update(cfg, state)
    assert state.shelf_melt_rate.numpy().max() == 4.0


def test_unknown_method_lists_the_available_laws(
    cfg_factory, state_factory, channel_factory
):
    cfg, state = _channel_state(
        cfg_factory, state_factory, channel_factory, method="meltnet"
    )
    with pytest.raises(ValueError, match=", ".join(available_melt_laws())):
        bmb.initialize(cfg, state)


def test_laws_that_need_the_ocean_say_so(cfg_factory, state_factory, channel_factory):
    cfg, state = _channel_state(
        cfg_factory, state_factory, channel_factory, method="pico"
    )
    del cfg.processes.ocean
    with pytest.raises(ValueError, match="'ocean' process"):
        bmb.initialize(cfg, state)


def test_needs_check_resolves_the_active_law(
    cfg_factory, state_factory, channel_factory
):
    cfg, state = _channel_state(
        cfg_factory, state_factory, channel_factory, method="pico"
    )
    bmb.initialize(cfg, state)
    del state.ocean_temp
    with pytest.raises(RuntimeError, match=r"bmb\(ocean_temp\)"):
        check_module_needs([bmb_package], state, cfg)
    assert _build_provider_map()["ocean_temp"] == ["ocean"]


def _flotation_thickness(topg):
    return -(1028.0 / 918.0) * topg


def test_partly_floating_cells_take_the_grounded_melt_of_grounded_ice(
    cfg_factory, state_factory, channel_factory
):
    """Under pmp, a cell mixes both melts by area; the enthalpy value at a
    floating node is not used, the grounded neighbours' is."""
    cfg = cfg_factory(
        prescribed={"form": "constant", "value": 5.0},
        grounding_line={"treatment": "pmp"},
    )
    thk, topg = channel_factory(front=32)
    grounded = thk > _flotation_thickness(topg)
    first = int(np.argmax(~grounded[0]))
    # A barely floating node next to a well grounded one: the grounding line
    # crosses the floating node's cell.
    thk[:, first - 1] = 1.5 * _flotation_thickness(topg[:, first - 1])
    thk[:, first] = 0.99 * _flotation_thickness(topg[:, first])
    state = state_factory(cfg, thk, topg)
    state.basal_melt_rate = tf.constant(
        np.where(grounded, GROUNDED_MELT, 99.0), tf.float32
    )
    _run(cfg, state)
    f = 1.0 - state.grounded_fraction.numpy()
    partial = ~grounded & (f < 1.0) & (thk > 0)
    assert partial[:, first].all()
    expected = -((1.0 - f) * GROUNDED_MELT + f * 5.0)
    np.testing.assert_allclose(state.bmb.numpy()[partial], expected[partial], rtol=1e-5)


def test_a_retreat_between_evaluations_keeps_the_ocean_melt(
    cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory(
        update_freq=10.0,
        prescribed={"form": "constant", "value": 5.0},
        grounding_line={"treatment": "fmp"},
    )
    thk, topg = channel_factory(front=32)
    first = int(np.argmax(thk[0] < _flotation_thickness(topg[0])))
    retreated = thk.copy()
    retreated[:, first - 1] = 0.9 * _flotation_thickness(topg[:, first - 1])

    fresh = _run(cfg, state_factory(cfg, retreated, topg))  # evaluated on it
    state = _run(cfg, state_factory(cfg, thk, topg))  # evaluated before it
    moved = state_factory(cfg, retreated, topg)
    for name in ("thk", "lsurf", "usurf"):
        setattr(state, name, getattr(moved, name))
    state.t.assign(1.0)
    bmb.update(cfg, state)  # not re-evaluated: t - tlast < update_freq
    assert (fresh.bmb.numpy()[:, first - 1] == -5.0).all()
    np.testing.assert_allclose(state.bmb.numpy(), fresh.bmb.numpy())
