"""Unit tests for the factorized spectral layers of the FNO architecture."""

from __future__ import annotations

import pytest
import tensorflow as tf

from igm.processes.iceflow.networks.architectures.nos import (
    FactorizedSpectralConv2D,
)


@pytest.mark.unit
def test_factorized_spectral_layer_has_finite_gradients_and_dynamic_shapes() -> None:
    layer = FactorizedSpectralConv2D(3, 4, modes1=3, modes2=3, rank=2)
    inputs = tf.random.stateless_normal((2, 3, 8, 10), seed=(5, 11))

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        output = layer(inputs)
        loss = tf.reduce_sum(tf.square(output))

    gradients = tape.gradient(loss, [inputs, *layer.trainable_variables])
    assert output.shape == (2, 4, 8, 10)
    assert layer(tf.zeros((1, 3, 10, 12))).shape == (1, 4, 10, 12)
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients
    )


@pytest.mark.unit
def test_factorized_spectral_layer_config_round_trip() -> None:
    layer = FactorizedSpectralConv2D(3, 4, modes1=5, modes2=6, rank=2)
    restored = FactorizedSpectralConv2D.from_config(layer.get_config())

    assert restored.in_channels == 3
    assert restored.out_channels == 4
    assert restored.modes1 == 5
    assert restored.modes2 == 6
    assert restored.rank == 2


@pytest.mark.unit
def test_factorized_spectral_layer_weights_round_trip(tmp_path) -> None:
    inputs = tf.random.stateless_normal((1, 3, 8, 10), seed=(7, 13))
    source = tf.keras.Sequential(
        [FactorizedSpectralConv2D(3, 4, modes1=3, modes2=3, rank=2)]
    )
    expected = source(inputs)
    checkpoint = tmp_path / "factorized.weights.h5"
    source.save_weights(checkpoint)

    restored = tf.keras.Sequential(
        [FactorizedSpectralConv2D(3, 4, modes1=3, modes2=3, rank=2)]
    )
    restored(inputs)
    restored.load_weights(checkpoint)

    tf.debugging.assert_near(restored(inputs), expected)
