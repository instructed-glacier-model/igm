"""Unit tests for the hybrid DahuNet + FNO emulator architecture."""

from __future__ import annotations

import pytest
import tensorflow as tf

from igm.processes.iceflow.networks.architectures import (
    Architectures,
    HybridDahuNetFNO,
)
from igm.processes.iceflow.networks.architectures.nos import (
    FactorizedSpectralConv2D,
    SpectralConv2D,
)

INPUTS = ["thk", "usurf", "arrhenius", "tau_ref", "dX", "water_level"]


@pytest.mark.unit
def test_hybrid_dahunet_fno_is_registered() -> None:
    assert Architectures["hybrid_dahunet_fno"] is HybridDahuNetFNO


@pytest.mark.unit
def test_hybrid_dahunet_fno_uses_cnn_dahunet_branch() -> None:
    model = HybridDahuNetFNO(
        input_names=INPUTS,
        Nz=1,
        network_params={
            "features": ["dsdx", "grad_s"],
            "nb_layers": 3,
            "width": 12,
            "modes1": 4,
            "modes2": 3,
        },
    )

    assert model.dahunet_branch.backend_name == "cnn"
    assert model.dahunet_branch.features == ["dsdx", "grad_s"]
    assert model.dahunet_branch.backend_params["nb_layers"] == 3
    assert model.fno_branch.width == 12
    assert model.fno_branch.modes1 == 4
    assert model.fno_branch.modes2 == 3


@pytest.mark.unit
def test_hybrid_dahunet_fno_output_is_branch_sum(monkeypatch) -> None:
    model = HybridDahuNetFNO(input_names=INPUTS, Nz=2)

    def constant_output(value):
        def call(inputs, training=False):
            shape = tf.concat([tf.shape(inputs)[:-1], [model.nb_outputs]], axis=0)
            return tf.fill(shape, tf.cast(value, inputs.dtype))

        return call

    monkeypatch.setattr(model.dahunet_branch, "call", constant_output(2.0))
    monkeypatch.setattr(model.fno_branch, "call", constant_output(3.0))

    inputs = tf.zeros((1, 8, 10, len(INPUTS)), dtype=tf.float32)
    output = model(inputs, training=False)

    assert output.shape == (1, 8, 10, 4)
    tf.debugging.assert_equal(output, tf.fill(output.shape, 5.0))


@pytest.mark.unit
def test_hybrid_dahunet_fno_factorized_spectral_branch_is_compact() -> None:
    compact = HybridDahuNetFNO(
        input_names=INPUTS,
        Nz=2,
        network_params={
            "width": 8,
            "modes1": 64,
            "modes2": 64,
            "fno_layers": 3,
            "factorization_rank": 7,
        },
    )
    dense_layer = SpectralConv2D(8, 8, 64, 64)
    compact_layer = FactorizedSpectralConv2D(8, 8, 64, 64, rank=7)
    inputs = tf.zeros((1, 8, 65, 130), dtype=tf.float32)

    assert dense_layer(inputs).shape == compact_layer(inputs).shape == inputs.shape
    assert compact.fno_branch.n_layers == 3
    assert compact.fno_branch.factorization_rank == 7

    # The fixed terms are the configured DahuNet branch and the FNO lift/head.
    # Each spectral block also has an 8x8 pointwise kernel and eight biases.
    dahunet_parameters = 32_236
    fno_lift_and_head_parameters = 1_740
    compact_total = (
        dahunet_parameters
        + fno_lift_and_head_parameters
        + 3 * (compact_layer.count_params() + 8 * 8 + 8)
    )
    dense_total = (
        dahunet_parameters
        + fno_lift_and_head_parameters
        + 4 * (dense_layer.count_params() + 8 * 8 + 8)
    )
    assert compact_total == 379_600
    assert dense_total == 4_228_568
    assert compact_total < dense_total / 11
    assert compact.resolved_params()["network_params"]["fno_layers"] == 3
