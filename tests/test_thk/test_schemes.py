#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for explicit thickness-backend dictionaries and dispatch."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk import fronts, transport
from igm.processes.thk import thk as thk_module


def _cfg(scheme):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": scheme,
                    "calving_front": False,
                    "ratio_density": 0.91,
                }
            }
        }
    )


def _state():
    return SimpleNamespace(
        thk=tf.ones((3, 4), dtype=tf.float32),
        topg=tf.zeros((3, 4), dtype=tf.float32),
        it=0,
    )


def test_builtin_schemes_expose_the_common_backend_interface():
    assert transport.available_transport_schemes() == (
        "adi",
        "explicit",
        "ffsl",
        "implicit",
        "implicit_x",
    )
    for name in transport.available_transport_schemes():
        # Without a calving front the scheme alone moves mass.
        resolved_name, module = transport.get_transport(_cfg(name))
        assert resolved_name == name
        assert callable(module.initialize)
        assert callable(module.update)


def test_root_package_keeps_only_the_process_lifecycle_public():
    import igm.processes.thk as thk_package

    assert callable(thk_package.initialize)
    assert callable(thk_package.update)
    assert callable(thk_package.finalize)
    assert not hasattr(thk_package, "TransportSchemes")
    assert not hasattr(thk_package, "FrontMethods")


def test_new_scheme_needs_only_one_dictionary_entry(monkeypatch):
    calls = []

    class TestScheme:
        @staticmethod
        def initialize(cfg, state):
            calls.append("initialize")
            state.test_scheme_initialized = True

        @staticmethod
        def update(cfg, state):
            calls.append("update")
            state.divflux = tf.zeros_like(state.thk)
            state.thk = state.thk + 2.0

    monkeypatch.setitem(
        transport.TransportSchemes, "test_scheme", TestScheme
    )

    cfg = _cfg("TEST_SCHEME")
    state = _state()
    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    assert calls == ["initialize", "update"]
    assert state.test_scheme_initialized
    tf.debugging.assert_equal(state.thk, tf.fill((3, 4), 3.0))
    tf.debugging.assert_equal(state.usurf, state.thk)


def test_unknown_scheme_error_lists_available_backends():
    with pytest.raises(
        ValueError,
        match="available schemes: adi, explicit, ffsl, implicit, implicit_x",
    ):
        thk_module.initialize(_cfg("does_not_exist"), _state())
