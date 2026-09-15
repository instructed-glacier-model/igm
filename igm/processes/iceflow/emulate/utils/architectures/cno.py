#!/usr/bin/env python3


"""Convolutional neural operator with GPU-native filtered activations.

Follows the authors' vanilla CNO (NeurIPS 2023), including residual encoder
skips and invariant blocks, without batch normalization. Bicubic interpolation
is an approximation to the paper's ideal sinc filters, as in the official
simplified implementation, not a claim of exact continuous-discrete equivalence.

The fixed, separable Keys cubic filters reproduce antialiased bicubic resize
at integer factors. Unlike ``tf.image.resize(..., antialias=True)``, they stay
on the GPU and preserve the model's floating-point dtype.

References:
    https://arxiv.org/abs/2302.01178
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_simplified
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .dahunet import FEATURES, _dict_proxies


def _cubic(x: np.ndarray) -> np.ndarray:
    """Keys cubic interpolation kernel (a = -1/2)."""
    x = np.abs(x)
    return np.where(
        x <= 1.0,
        1.5 * x**3 - 2.5 * x**2 + 1.0,
        np.where(x < 2.0, -0.5 * x**3 + 2.5 * x**2 - 4.0 * x + 2.0, 0.0),
    )


def _filter_axis(
    x: tf.Tensor, weights: np.ndarray, axis: int, stride: int
) -> tf.Tensor:
    """Separable depthwise filtering, renormalized at domain boundaries."""
    channels = int(x.shape[-1])
    taps, phases = weights.shape
    if axis == 1:
        x = tf.transpose(x, [0, 2, 1, 3])
    shape = tf.shape(x)
    # Depthwise kernels require equal spatial strides on CPU. Treat each row
    # as an independent batch so this also supports strided 1D GPU filtering.
    rows = tf.reshape(x, [shape[0] * shape[1], 1, shape[2], channels])
    kernel = tf.constant(weights.reshape(1, taps, 1, phases), dtype=x.dtype)
    strides = [1, stride, stride, 1]
    filtered = tf.nn.depthwise_conv2d(
        rows, tf.tile(kernel, [1, 1, channels, 1]), strides, padding="SAME"
    )
    # Only a 1D signal of ones is needed, rather than a full-grid/channel mask.
    length = shape[2]
    mask_shape = [1, 1, length, 1]
    norm = tf.nn.depthwise_conv2d(
        tf.ones(mask_shape, dtype=x.dtype), kernel, strides, padding="SAME"
    )
    if phases == 1:
        filtered = tf.reshape(
            filtered / norm, [shape[0], shape[1], length // stride, channels]
        )
    else:
        # Interleave phases along space, without inserting a zero-filled grid.
        filtered = tf.reshape(filtered, [shape[0], shape[1], length, channels, phases])
        filtered = tf.transpose(filtered, [0, 1, 2, 4, 3])
        filtered = tf.reshape(filtered, [shape[0], shape[1], length * phases, channels])
        norm = tf.reshape(norm, [1, 1, length * phases, 1])
        filtered = filtered / norm
    return tf.transpose(filtered, [0, 2, 1, 3]) if axis == 1 else filtered


def _upsample2(x: tf.Tensor) -> tf.Tensor:
    positions = np.arange(-2, 3, dtype=np.float64)[:, None]
    weights = _cubic(positions - np.array([-0.25, 0.25])[None, :])
    x = _filter_axis(x, weights, axis=2, stride=1)
    return _filter_axis(x, weights, axis=1, stride=1)


def _downsample(x: tf.Tensor, factor: int) -> tf.Tensor:
    positions = np.arange(4 * factor, dtype=np.float64)
    weights = _cubic((positions - (4 * factor - 1) / 2.0) / factor)[:, None]
    x = _filter_axis(x, weights, axis=2, stride=factor)
    return _filter_axis(x, weights, axis=1, stride=factor)


class _FilteredActivation(tf.keras.layers.Layer):
    """Upsample -> leaky ReLU -> low-pass and resample to the target band."""

    def __init__(self, scale: str = "same", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.nn.leaky_relu(_upsample2(x), alpha=0.01)
        if self.scale == "up":
            return x
        return _downsample(x, 4 if self.scale == "down" else 2)


class _CNOBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        scale: str = "same",
        normalization: str = "none",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            channels, 3, padding="same", dtype=self.dtype_policy
        )
        self.activation = _FilteredActivation(scale, dtype=self.dtype_policy)
        self.normalizer = (
            tf.keras.layers.GroupNormalization(
                groups=-1, epsilon=1e-5, dtype=self.dtype_policy
            )
            if normalization == "instance"
            else None
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return self.activation(x)


class _ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, normalization: str = "none", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.block = _CNOBlock(
            channels, normalization=normalization, dtype=self.dtype_policy
        )
        self.conv = tf.keras.layers.Conv2D(
            channels, 3, padding="same", dtype=self.dtype_policy
        )
        self.normalizer = (
            tf.keras.layers.GroupNormalization(
                groups=-1, epsilon=1e-5, dtype=self.dtype_policy
            )
            if normalization == "instance"
            else None
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = self.conv(self.block(x))
        if self.normalizer is not None:
            residual = self.normalizer(residual)
        return x + residual


class _LiftProject(tf.keras.layers.Layer):
    def __init__(self, channels: int, latent_channels: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.block = _CNOBlock(latent_channels, dtype=self.dtype_policy)
        self.conv = tf.keras.layers.Conv2D(
            channels, 3, padding="same", dtype=self.dtype_policy
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.conv(self.block(x))


class CNO(tf.keras.Model):
    """Vanilla operator U-Net, with optional DahuNet physics-derived features.

    ``n_layers`` counts downsampling stages (and matching upsampling stages).
    Channel counts are ``[multiplier/2, multiplier, 2*multiplier, ...]``.
    Arbitrary rectangular grids are padded and cropped back, allowing exact
    factor-two interpolation at every stage. ``input_downsampling`` optionally
    restricts the operator to a coarser input/output band; a hybrid's CNN branch
    remains at full resolution to represent fine-scale corrections. Optional
    ``normalization: instance`` normalizes spatially per channel without
    moving statistics (equivalent to training-mode batch normalization for
    one full-domain sample). Lift/projection blocks stay unnormalized.
    ``max_channels`` caps channel growth in deep pyramids, and optional RMS
    feature normalization balances the dimensionless physics features without
    using velocity observations. Both options are disabled by default.
    """

    _DEFAULTS = {
        "channel_multiplier": (16, int),
        "n_layers": (3, int),
        "n_res": (1, int),
        "n_res_neck": (2, int),
        "latent_channels": (32, int),
        "features": ((), tuple),
        "use_grid": (True, bool),
        "input_downsampling": (1, int),
        "normalization": ("none", str),
        "max_channels": (0, int),
        "feature_normalization": ("none", str),
    }

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_names = [str(name) for name in input_names]
        self.Nz = int(Nz)
        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz
        self.input_normalizer = None
        params = dict(network_params or {})
        unexpected = sorted(set(params) - set(self._DEFAULTS))
        if unexpected:
            raise ValueError(f"Unexpected keys in network_params: {unexpected}")
        for key, (default, kind) in self._DEFAULTS.items():
            setattr(self, key, kind(params.get(key, default)))
        if self.Nz < 1 or self.nb_inputs < 1:
            raise ValueError("Nz and the number of inputs must be positive")
        if self.channel_multiplier < 2 or self.channel_multiplier % 2:
            raise ValueError("channel_multiplier must be a positive even integer >= 2")
        if self.n_layers < 1 or self.latent_channels < 1:
            raise ValueError("n_layers and latent_channels must be positive")
        if self.n_res < 0 or self.n_res_neck < 0:
            raise ValueError("Residual block counts must be nonnegative")
        if self.input_downsampling not in (1, 2, 4, 8):
            raise ValueError("input_downsampling must be 1, 2, 4 or 8")
        if self.normalization not in ("none", "instance"):
            raise ValueError("normalization must be 'none' or 'instance'")
        if self.max_channels and self.max_channels < self.channel_multiplier:
            raise ValueError("max_channels must be zero or >= channel_multiplier")
        if self.feature_normalization not in ("none", "rms"):
            raise ValueError("feature_normalization must be 'none' or 'rms'")
        unknown = sorted(set(self.features) - set(FEATURES))
        if unknown:
            raise ValueError(f"Unknown physics features: {unknown}")
        self.feature_indices = None
        if self.features:
            required = ["thk", "usurf", "dX"]
            if not all(name in self.input_names for name in required):
                raise ValueError(
                    "Physics features require thk, usurf and dX in input_names"
                )
            self.feature_indices = tuple(
                self.input_names.index(name) for name in required
            )

        channels = [self.channel_multiplier // 2] + [
            (
                min(self.channel_multiplier * 2**i, self.max_channels)
                if self.max_channels
                else self.channel_multiplier * 2**i
            )
            for i in range(self.n_layers)
        ]
        layer_kwargs = {"dtype": self.dtype_policy}
        block_kwargs = {**layer_kwargs, "normalization": self.normalization}
        self.lift = _LiftProject(channels[0], self.latent_channels, **layer_kwargs)
        self.encoder = [
            _CNOBlock(channels[i + 1], "down", **block_kwargs)
            for i in range(self.n_layers)
        ]
        self.residuals = [
            [_ResidualBlock(channels[i], **block_kwargs) for _ in range(self.n_res)]
            for i in range(self.n_layers)
        ]
        self.neck = [
            _ResidualBlock(channels[-1], **block_kwargs) for _ in range(self.n_res_neck)
        ]
        self.invariant = [_CNOBlock(c, **block_kwargs) for c in channels]
        self.decoder = [
            _CNOBlock(channels[i], "up", **block_kwargs)
            for i in reversed(range(self.n_layers))
        ]
        self.project = _LiftProject(
            self.nb_outputs, self.latent_channels, **layer_kwargs
        )

    def resolved_params(self) -> dict[str, Any]:
        params = {
            key: kind(getattr(self, key)) for key, (_, kind) in self._DEFAULTS.items()
        }
        params["features"] = [str(name) for name in self.features]
        return {
            "input_names": list(self.input_names),
            "Nz": self.Nz,
            "network_params": params,
        }

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(self.resolved_params())
        return config

    def build(self, input_shape) -> None:
        if self.built:
            return
        shape = tf.TensorShape(input_shape)
        if shape.rank != 4 or (
            shape[-1] is not None and int(shape[-1]) != self.nb_inputs
        ):
            raise ValueError(f"CNO expects [B, H, W, {self.nb_inputs}], got {shape}")
        # Build weights on a small grid; never allocate a full-resolution dummy.
        size = 2**self.n_layers * self.input_downsampling
        dummy = tf.ones((1, size, size, self.nb_inputs), dtype=self.compute_dtype)
        self.call(dummy)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        raw = tf.cast(inputs, self.compute_dtype)
        x = (
            raw
            if self.input_normalizer is None
            else self.input_normalizer(raw, training=training)
        )
        if self.feature_indices is not None:
            proxies = _dict_proxies(raw, *self.feature_indices)
            features = tf.concat(
                [FEATURES[name](proxies) for name in self.features], axis=-1
            )
            if self.feature_normalization == "rms":
                mean_square = tf.reduce_mean(features**2, axis=[1, 2], keepdims=True)
                features *= tf.math.rsqrt(
                    tf.maximum(mean_square, tf.cast(1e-12, features.dtype))
                )
            x = tf.concat([x, features], axis=-1)
        shape = tf.shape(x)
        if self.use_grid:
            y, z = tf.meshgrid(
                tf.linspace(tf.cast(0.0, x.dtype), tf.cast(1.0, x.dtype), shape[1]),
                tf.linspace(tf.cast(0.0, x.dtype), tf.cast(1.0, x.dtype), shape[2]),
                indexing="ij",
            )
            grid = tf.broadcast_to(
                tf.stack([y, z], axis=-1)[None], [shape[0], shape[1], shape[2], 2]
            )
            x = tf.concat([x, grid], axis=-1)
        multiple = 2**self.n_layers * self.input_downsampling
        pad_y, pad_x = (-shape[1]) % multiple, (-shape[2]) % multiple
        x = tf.pad(x, [[0, 0], [0, pad_y], [0, pad_x], [0, 0]])
        if self.input_downsampling > 1:
            x = _downsample(x, self.input_downsampling)
        x = self.lift(x)
        skips = []
        for blocks, encoder in zip(self.residuals, self.encoder):
            skip = x
            for block in blocks:
                skip = block(skip)
            skips.append(skip)
            x = encoder(x)
        for block in self.neck:
            x = block(x)
        x = self.invariant[-1](x)
        for i, decoder in enumerate(self.decoder):
            level = self.n_layers - i
            if i:
                x = tf.concat([x, self.invariant[level](skips[level])], axis=-1)
            x = decoder(x)
        x = tf.concat([x, self.invariant[0](skips[0])], axis=-1)
        x = self.project(x)
        for _ in range(self.input_downsampling.bit_length() - 1):
            x = _upsample2(x)
        return x[:, : shape[1], : shape[2], :]
