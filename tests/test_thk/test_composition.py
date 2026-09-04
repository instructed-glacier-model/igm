#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for modular composition of transport and front evolution."""

from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk import fronts, transport
from igm.processes.thk import thk as thk_module


def _module(name, **attributes):
    """Build a module-shaped backend for dispatch tests."""
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _cfg(scheme, method="test_front"):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": scheme,
                    "calving_front": True,
                    "method": method,
                    "ratio_density": 0.91,
                    "boundary": {
                        "left": "zero",
                        "right": "zero",
                        "top": "zero",
                        "bottom": "zero",
                    },
                }
            }
        }
    )


def _state():
    thickness = tf.ones((3, 4), tf.float32)
    return SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        it=0,
    )


def test_after_transport_front_composes_without_changing_thk_dispatch(monkeypatch):
    calls = []

    def initialize_scheme(cfg, state):
        calls.append("scheme.initialize")

    def update_scheme(cfg, state):
        calls.append("scheme.update")
        state.thk = state.thk + 2.0
        state.divflux = tf.zeros_like(state.thk)

    def finalize_scheme(cfg, state):
        calls.append("scheme.finalize")

    def initialize_front(cfg, state):
        calls.append("front.initialize")

    def update_front(cfg, state):
        calls.append("front.update")
        state.thk = state.thk - 0.5

    def finalize_front(cfg, state):
        calls.append("front.finalize")

    test_scheme = _module(
        "test_scheme",
        SUPPORTED_BOUNDARY_MODES=("zero",),
        initialize=initialize_scheme,
        update=update_scheme,
        finalize=finalize_scheme,
    )
    test_front = _module(
        "test_front",
        SUPPORTED_BOUNDARY_MODES=("zero",),
        UPDATE_MODE="after_transport",
        COMPATIBLE_TRANSPORTS=("test_scheme",),
        AVAILABLE=True,
        UNAVAILABLE_REASON="",
        initialize=initialize_front,
        update=update_front,
        finalize=finalize_front,
    )

    monkeypatch.setitem(
        transport.TransportSchemes, "test_scheme", test_scheme
    )
    monkeypatch.setitem(fronts.FrontMethods, "test_front", test_front)

    state = _state()
    cfg = _cfg("test_scheme")
    thk_module.initialize(cfg, state)
    assert state.thk_components.transport is test_scheme
    assert state.thk_components.front is test_front
    assert state.thk_components.pipeline == (test_scheme, test_front)
    thk_module.update(cfg, state)
    thk_module.finalize(cfg, state)

    assert calls == [
        "scheme.initialize",
        "front.initialize",
        "scheme.update",
        "front.update",
        "front.finalize",
        "scheme.finalize",
    ]
    tf.debugging.assert_near(state.thk, tf.fill((3, 4), 2.5))


def test_transport_owning_front_does_not_initialize_unused_transport(monkeypatch):
    calls = []

    def initialize_transport(cfg, state):
        calls.append("transport.initialize")

    def update_transport(cfg, state):
        calls.append("transport.update")

    def finalize_transport(cfg, state):
        calls.append("transport.finalize")

    def initialize_front(cfg, state):
        calls.append("front.initialize")

    def update_front(cfg, state):
        calls.append("front.update")
        state.divflux = tf.zeros_like(state.thk)

    def finalize_front(cfg, state):
        calls.append("front.finalize")

    unused_transport = _module(
        "unused",
        SUPPORTED_BOUNDARY_MODES=("zero",),
        initialize=initialize_transport,
        update=update_transport,
        finalize=finalize_transport,
    )
    owning_front = _module(
        "owning",
        SUPPORTED_BOUNDARY_MODES=("zero",),
        UPDATE_MODE="replace_transport",
        COMPATIBLE_TRANSPORTS=("unused",),
        AVAILABLE=True,
        UNAVAILABLE_REASON="",
        initialize=initialize_front,
        update=update_front,
        finalize=finalize_front,
    )
    monkeypatch.setitem(transport.TransportSchemes, "unused", unused_transport)
    monkeypatch.setitem(fronts.FrontMethods, "owning", owning_front)

    cfg = _cfg("unused", method="owning")
    state = _state()
    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)
    thk_module.finalize(cfg, state)

    assert calls == ["front.initialize", "front.update", "front.finalize"]


def test_transport_owning_subgrid_front_rejects_ignored_scheme(monkeypatch):
    def initialize(cfg, state):
        pass

    def update(cfg, state):
        pass

    monkeypatch.setitem(
        transport.TransportSchemes,
        "other",
        _module("other", initialize=initialize, update=update),
    )

    with pytest.raises(ValueError, match="replace_transport.*explicit"):
        thk_module.initialize(_cfg("other", method="sub_grid"), _state())


def test_unavailable_front_is_reported_by_the_dictionary():
    assert fronts.available_front_methods() == ("sub_grid",)
    with pytest.raises(ValueError, match="level_set.*unavailable"):
        thk_module.initialize(_cfg("explicit", method="level_set"), _state())


def test_level_set_bookkeeping_is_namespaced():
    state = _state()
    state.thk_components = SimpleNamespace(component_state={})

    fronts.level_set.initialize(_cfg("explicit", method="level_set"), state)

    assert state.thk_components.component_state["level_set"] == {
        "steps_since_reinit": 0,
        "psi_built": False,
    }
