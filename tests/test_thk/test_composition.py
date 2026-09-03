#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for modular composition of transport and front evolution."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk import fronts, transport
from igm.processes.thk import thk as thk_module
from igm.processes.thk.fronts import FrontMethod


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

    class TestScheme:
        SUPPORTED_BOUNDARY_MODES = ("zero",)

        @staticmethod
        def initialize(cfg, state):
            calls.append("scheme.initialize")

        @staticmethod
        def update(cfg, state):
            calls.append("scheme.update")
            state.thk = state.thk + 2.0
            state.divflux = tf.zeros_like(state.thk)

    class TestFront:
        SUPPORTED_BOUNDARY_MODES = ("zero",)

        @staticmethod
        def initialize(cfg, state):
            calls.append("front.initialize")

        @staticmethod
        def update(cfg, state):
            calls.append("front.update")
            state.thk = state.thk - 0.5

    monkeypatch.setitem(
        transport.TransportSchemes, "test_scheme", TestScheme
    )
    monkeypatch.setitem(
        fronts.FrontMethods,
        "test_front",
        FrontMethod(
            backend=TestFront,
            update_mode="after_transport",
            compatible_transports=("test_scheme",),
            available=True,
            unavailable_reason="",
        ),
    )

    state = _state()
    cfg = _cfg("test_scheme")
    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    assert calls == [
        "scheme.initialize",
        "front.initialize",
        "scheme.update",
        "front.update",
    ]
    tf.debugging.assert_near(state.thk, tf.fill((3, 4), 2.5))


def test_transport_owning_front_does_not_initialize_unused_transport(monkeypatch):
    calls = []

    class UnusedTransport:
        SUPPORTED_BOUNDARY_MODES = ("zero",)

        @staticmethod
        def initialize(cfg, state):
            calls.append("transport.initialize")

        @staticmethod
        def update(cfg, state):
            calls.append("transport.update")

        @staticmethod
        def finalize(cfg, state):
            calls.append("transport.finalize")

    class OwningFront:
        SUPPORTED_BOUNDARY_MODES = ("zero",)

        @staticmethod
        def initialize(cfg, state):
            calls.append("front.initialize")

        @staticmethod
        def update(cfg, state):
            calls.append("front.update")
            state.divflux = tf.zeros_like(state.thk)

        @staticmethod
        def finalize(cfg, state):
            calls.append("front.finalize")

    monkeypatch.setitem(transport.TransportSchemes, "unused", UnusedTransport)
    monkeypatch.setitem(
        fronts.FrontMethods,
        "owning",
        FrontMethod(
            backend=OwningFront,
            update_mode="replace_transport",
            compatible_transports=("unused",),
            available=True,
            unavailable_reason="",
        ),
    )

    cfg = _cfg("unused", method="owning")
    state = _state()
    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)
    thk_module.finalize(cfg, state)

    assert calls == ["front.initialize", "front.update", "front.finalize"]


def test_transport_owning_subgrid_front_rejects_ignored_scheme(monkeypatch):
    class OtherTransport:
        @staticmethod
        def initialize(cfg, state):
            pass

        @staticmethod
        def update(cfg, state):
            pass

    monkeypatch.setitem(transport.TransportSchemes, "other", OtherTransport)

    with pytest.raises(ValueError, match="replace_transport.*explicit"):
        thk_module.initialize(_cfg("other", method="sub_grid"), _state())


def test_unavailable_front_is_reported_by_the_dictionary():
    with pytest.raises(ValueError, match="level_set.*unavailable"):
        thk_module.initialize(_cfg("explicit", method="level_set"), _state())
