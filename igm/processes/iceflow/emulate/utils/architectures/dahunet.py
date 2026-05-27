"""
DahuNet — unified physics-informed emulator for ice flow.

Features are selected from FEATURES at construction time.
Under tf.function the feature list is traced once and compiled into the graph,
so there is zero Python overhead at inference time vs. hard-coded features.

Backends: "cnn" (default, DahuNet_5-style), "fno" (FNO2-style), "mlp".

Feature scrambling
------------------
Set ``scramble_features=True`` to apply a *fixed* Fourier phase scramble to
the computed physics features before they are concatenated with the raw inputs.
Raw inputs are never modified.

Design rationale
~~~~~~~~~~~~~~~~
The scramble is fixed: the same phase matrix is applied at every forward call
throughout the entire simulation.  This is the only fair choice for the
online warm-start setting studied in the paper, for two reasons:

1. A fresh scramble at each call (or each retraining event) would additionally
   break the temporal coherence of the feature representation — the very
   property that the warm-start argument says is central.  Worse performance
   would then be overdetermined and uninterpretable.

2. A fixed scramble makes the test conservative: the network is free to
   partially adapt its weights to the fixed distortion over successive
   retraining events.  If performance is still worse, spatial structure
   demonstrably matters.  If performance is comparable, the value
   distribution of the features is sufficient and their spatial arrangement
   is not — a clean scientific conclusion either way.

Implementation: ``tf.random.stateless_uniform`` with a fixed integer seed
guarantees that the same phase tensor is produced at every call for a given
feature shape, without any mutable generator state.

Usage
~~~~~
    # baseline
    model = DahuNet(cfg, ..., features=("dsdx", "dsdy", "u_sia"))

    # null hypothesis: same features, spatial physics destroyed
    null  = DahuNet(cfg, ..., features=("dsdx", "dsdy", "u_sia"),
                    scramble_features=True, scramble_seed=42)

    # Interpretation:
    #   worse null  → spatial arrangement of features carries physical info
    #   similar     → value distribution is sufficient; gain is expressivity
"""

import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Sequence

from .nos import SpectralConv2D

# ─────────────────────────────────────────────────────────────────────────────
# Physics feature catalogue
# ─────────────────────────────────────────────────────────────────────────────

SCALE_DSDX = 0.1  # (-)
SCALE_THK_GLACIER = 200.0  # (m)
SCALE_THK_SHELF = 100.0  # (m)

FEATURES = {
    "dsdx": lambda p: p["dsdx"] / SCALE_DSDX,
    "dsdy": lambda p: p["dsdy"] / SCALE_DSDX,
    "grad_s": lambda p: p["grad_s"] / SCALE_DSDX,
    "u_sia": lambda p: tf.math.log(
        (p["grad_s"] / SCALE_DSDX) ** 3.0 * (p["thk"] / SCALE_THK_GLACIER) ** 4.0 + 1.0
    ),
    "u_weertman": lambda p: tf.math.log(
        (p["grad_s"] / SCALE_DSDX) ** 3.0 * (p["thk"] / SCALE_THK_GLACIER) ** 3.0 + 1.0
    ),
    "u_float": lambda p: tf.math.log(((p["thk"] / SCALE_THK_SHELF) ** 3.0 + 1.0)),
}

FEATURES_DEFAULT = ("dsdx", "dsdy", "grad_s", "u_sia", "u_weertman")


def _dict_proxies(
    raw: tf.Tensor,
    idx_thk: int,
    idx_usurf: int,
    idx_dX: int,
) -> Dict[str, tf.Tensor]:
    """Compute shared base quantities once per forward pass."""
    thk = raw[..., idx_thk : idx_thk + 1]
    usurf = raw[..., idx_usurf : idx_usurf + 1]
    dX = raw[..., idx_dX : idx_dX + 1]
    inv2 = 1.0 / (2.0 * dX + 1e-10)
    dsdx = (tf.roll(usurf, -1, axis=2) - tf.roll(usurf, 1, axis=2)) * inv2
    dsdy = (tf.roll(usurf, -1, axis=1) - tf.roll(usurf, 1, axis=1)) * inv2
    grad_s = tf.sqrt(dsdx**2 + dsdy**2 + 1e-20)
    return dict(thk=thk, usurf=usurf, dX=dX, dsdx=dsdx, dsdy=dsdy, grad_s=grad_s)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed Fourier phase scrambler
# ─────────────────────────────────────────────────────────────────────────────


class _FourierScramble(tf.keras.layers.Layer):
    """
    Apply a *fixed* Fourier phase scramble to every channel of [B, H, W, C].

    The phase matrix is determined entirely by ``seed`` and the spatial shape
    of the input.  Because ``tf.random.stateless_uniform`` is used, the
    identical phase tensor is produced on every call — no mutable RNG state.

    Consequence: the network sees exactly the same spatial distortion at every
    forward pass, at every time step, at every retraining event.  The scramble
    is a permanent but consistent remapping of feature space, not a source of
    stochastic noise.

    Parameters
    ----------
    seed : int
        Determines the phase matrix.  Different seeds give different scrambles;
        running several seeds and averaging results gives a more stable null
        estimate.
    n_features : int
        Number of feature channels (C).  Must be known statically so the phase
        tensor shape is fully determined by seed + (C, H, W).
    """

    def __init__(self, seed: int, n_features: int, **kwargs: Any) -> None:
        super().__init__(trainable=False, **kwargs)
        self._seed = seed
        self._n_features = n_features
        # Stateless seed pair: counter fixed at 0 → always same draw.
        self._stateless_seed = tf.constant([seed, 0], dtype=tf.int32)

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        features : [B, H, W, C]
        Returns  : [B, H, W, C] — phases scrambled, amplitudes preserved.
        """
        x = tf.cast(features, tf.float32)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self._n_features  # static
        W_half = W // 2 + 1

        # ── fixed phase matrix [C, H, W_half] ────────────────────────────────
        # stateless_uniform: same (seed, shape) → same tensor, always.
        phases = tf.random.stateless_uniform(
            shape=[C, H, W_half],
            seed=self._stateless_seed,
            minval=-np.pi,
            maxval=np.pi,
        )

        # ── forward FFT: fold C into batch dim → [B*C, H, W_half] ────────────
        x_bchw = tf.transpose(x, [0, 3, 1, 2])  # [B, C, H, W]
        flat = tf.reshape(x_bchw, [B * C, H, W])
        spectrum = tf.signal.rfft2d(flat)  # [B*C, H, W_half]
        amplitude = tf.abs(spectrum)

        # ── broadcast phases over batch → [B*C, H, W_half] ───────────────────
        phases_bc = tf.tile(phases[tf.newaxis], [B, 1, 1, 1])  # [B, C, H, W_half]
        phases_flat = tf.reshape(phases_bc, [B * C, H, W_half])

        # ── replace phases, keep amplitudes ───────────────────────────────────
        new_spectrum = tf.cast(amplitude, tf.complex64) * tf.exp(
            tf.complex(tf.zeros_like(phases_flat), phases_flat)
        )

        # ── inverse FFT → restore [B, H, W, C] ───────────────────────────────
        scrambled_flat = tf.signal.irfft2d(new_spectrum, fft_length=[H, W])
        return tf.transpose(tf.reshape(scrambled_flat, [B, C, H, W]), [0, 2, 3, 1])

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "seed": self._seed,
            "n_features": self._n_features,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────


class _CNNBackend(tf.keras.layers.Layer):
    """DahuNet_5-style CNN: Dense lift → Conv2D stack with skip → 1×1 output."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_filters: int,
        n_layers: int,
        ker_size: int,
        use_residual: bool,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.use_residual = use_residual
        Conv = tf.keras.layers.Conv2D
        self.combine = tf.keras.layers.Dense(n_filters, dtype=tf.float32)
        self.backbone = [
            Conv(n_filters, ker_size, padding="same", dtype=tf.float32, name=f"bb_{i}")
            for i in range(n_layers)
        ]
        self.skip_proj = Conv(n_filters, 1, padding="same", dtype=tf.float32)
        self.out = Conv(n_out, 1, padding="same", dtype=tf.float32)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.nn.gelu(self.combine(x))
        skip = self.skip_proj(x)
        for i, conv in enumerate(self.backbone):
            res = x
            x = tf.nn.gelu(conv(x))
            if self.use_residual and i % 2 == 1:
                x = x + res
        return self.out(x + skip)


class _FNOBackend(tf.keras.layers.Layer):
    """FNO2-style spectral backbone: 4 Fourier blocks + MLP head."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        width: int,
        modes1: int,
        modes2: int,
        padding: int,
        use_grid: bool,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.padding = padding
        self.use_grid = use_grid
        Conv1 = lambda: tf.keras.layers.Conv2D(
            width, 1, data_format="channels_first", use_bias=True
        )
        self.fc0 = tf.keras.layers.Dense(width, dtype=tf.float32)
        self.convs = [SpectralConv2D(width, width, modes1, modes2) for _ in range(4)]
        self.ws = [Conv1() for _ in range(4)]
        self.fc1 = tf.keras.layers.Dense(128, dtype=tf.float32)
        self.fc2 = tf.keras.layers.Dense(n_out, dtype=tf.float32)

    def _grid(self, x: tf.Tensor) -> tf.Tensor:
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        gx = tf.tile(tf.reshape(tf.linspace(0.0, 1.0, H), [1, H, 1, 1]), [B, 1, W, 1])
        gy = tf.tile(tf.reshape(tf.linspace(0.0, 1.0, W), [1, 1, W, 1]), [B, H, 1, 1])
        return tf.concat([gx, gy], axis=-1)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.use_grid:
            x = tf.concat([x, self._grid(x)], axis=-1)
        x = self.fc0(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        if self.padding > 0:
            x = tf.pad(x, [[0, 0], [0, 0], [0, self.padding], [0, self.padding]])
        H_pad, W_pad = tf.shape(x)[2], tf.shape(x)[3]
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x1, x2 = conv(x), w(x)
            x = tf.nn.gelu(x1 + x2) if i < 3 else x1 + x2
        if self.padding > 0:
            x = x[:, :, : H_pad - self.padding, : W_pad - self.padding]
        x = tf.transpose(x, [0, 2, 3, 1])
        return self.fc2(tf.nn.gelu(self.fc1(x)))


class _MLPBackend(tf.keras.layers.Layer):
    """Per-pixel MLP (no spatial context — use for ablations)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_filters: int,
        n_layers: int,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.hidden = [
            tf.keras.layers.Dense(n_filters, activation="gelu", dtype=tf.float32)
            for _ in range(n_layers)
        ]
        self.out = tf.keras.layers.Dense(n_out, dtype=tf.float32)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# DahuNet
# ─────────────────────────────────────────────────────────────────────────────


class DahuNet(tf.keras.Model):
    """
    Unified physics-informed ice-flow emulator.

    Parameters
    ----------
    features : sequence of str
        Keys from FEATURES.  Fixed at construction.
    backend : {"cnn", "fno", "mlp"}
        Spatial backbone.
    scramble_features : bool
        If True, apply a fixed Fourier phase scramble to the computed physics
        features before concatenating them with the raw inputs.  The scramble
        is identical at every forward call throughout the simulation.
        See module docstring for the full design rationale.
    scramble_seed : int or None
        Seed that determines the fixed phase matrix.  Required when
        ``scramble_features=True``; ignored otherwise.  Different seeds give
        different (but equally valid) scrambles; running several seeds and
        comparing results gives a more stable null estimate.
    """

    def __init__(
        self,
        cfg: Any,
        nb_inputs: int,
        nb_outputs: int,
        input_normalizer: Optional[tf.keras.layers.Layer] = None,
        features: Sequence[str] = FEATURES_DEFAULT,
        backend: str = "cnn",
        scramble_features: bool = False,
        scramble_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if scramble_features and scramble_seed is None:
            raise ValueError(
                "scramble_seed must be an integer when scramble_features=True. "
                "The scramble is fixed for the lifetime of the model; the seed "
                "determines which fixed scramble is used."
            )

        cfg_unified = cfg.processes.iceflow.unified
        cfg_network = cfg.processes.iceflow.emulator.network

        self.input_normalizer = input_normalizer
        self.scramble_features = scramble_features

        # Channel indices
        names = list(cfg_unified.inputs)
        self.idx_thk = names.index("thk")
        self.idx_usurf = names.index("usurf")
        self.idx_dX = names.index("dX")

        # Validate & resolve feature functions
        unknown = [f for f in features if f not in FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown features: {unknown}\nAvailable: {sorted(FEATURES)}"
            )
        self.feature_fns = [FEATURES[f] for f in features]
        n_features = len(features)

        # Fixed scrambler (None when disabled)
        self._scrambler = (
            _FourierScramble(
                seed=scramble_seed,
                n_features=n_features,
                name="feature_scrambler",
            )
            if scramble_features
            else None
        )

        n_combined = nb_inputs + n_features

        # Instantiate backend
        if backend == "cnn":
            self._backend = _CNNBackend(
                n_combined,
                nb_outputs,
                n_filters=int(getattr(cfg_network, "nb_out_filter", 32)),
                n_layers=int(getattr(cfg_network, "nb_layers", 16)),
                ker_size=int(getattr(cfg_network, "conv_ker_size", 3)),
                use_residual=bool(getattr(cfg_network, "residual", True)),
            )
            dummy_H, dummy_W = 8, 8

        elif backend == "fno":
            cfg_net = cfg_unified.network
            modes1 = int(getattr(cfg_net, "modes1", 8))
            modes2 = int(getattr(cfg_net, "modes2", modes1))
            self._backend = _FNOBackend(
                n_combined,
                nb_outputs,
                width=int(getattr(cfg_net, "width", 32)),
                modes1=modes1,
                modes2=modes2,
                padding=int(getattr(cfg_net, "padding", 9)),
                use_grid=bool(getattr(cfg_net, "use_grid", True)),
            )
            dummy_H = max(16, modes1 + 1)
            dummy_W = max(16, 2 * modes2 + 2)

        elif backend == "mlp":
            self._backend = _MLPBackend(
                n_combined,
                nb_outputs,
                n_filters=int(getattr(cfg_network, "nb_out_filter", 64)),
                n_layers=int(getattr(cfg_network, "nb_layers", 8)),
            )
            dummy_H, dummy_W = 8, 8

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose: cnn, fno, mlp.")

        # Build weights
        self(tf.zeros((1, dummy_H, dummy_W, nb_inputs), tf.float32), training=False)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        raw = tf.cast(inputs, tf.float32)
        x = raw

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        proxies = _dict_proxies(raw, self.idx_thk, self.idx_usurf, self.idx_dX)
        features = tf.concat([fn(proxies) for fn in self.feature_fns], axis=-1)

        if self._scrambler is not None:
            features = self._scrambler(features, training=training)

        return self._backend(tf.concat([features, x], axis=-1), training=training)