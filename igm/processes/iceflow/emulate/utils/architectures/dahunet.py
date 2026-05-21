"""
DahuNet — unified physics-informed emulator for ice flow.

Features are selected from FEATURES at construction time.
Under tf.function the feature list is traced once and compiled into the graph,
so there is zero Python overhead at inference time vs. hard-coded features.

Backends: "cnn" (default, DahuNet_5-style), "fno" (FNO2-style), "mlp".

"""

import tensorflow as tf
from typing import Any, Dict

from .nos import SpectralConv2D


# ─────────────────────────────────────────────────────────────────────────────
# Physics feature catalogue
#
# Each entry:  name -> callable(ctx) -> Tensor[B, H, W, 1]
#
# ctx keys (all [B, H, W, 1], float32):
#   thk, usurf, dX  — raw channel slices
#   dsdx, dsdy      — central-difference surface gradients
#   grad_s          — gradient magnitude  √(dsdx²+dsdy²)
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
        self.combine = tf.keras.layers.Dense(n_filters, dtype=self.compute_dtype)
        self.backbone = [
            Conv(n_filters, ker_size, padding="same", dtype=self.compute_dtype, name=f"bb_{i}")
            for i in range(n_layers)
        ]
        self.skip_proj = Conv(n_filters, 1, padding="same", dtype=self.compute_dtype)
        self.out = Conv(n_out, 1, padding="same", dtype=self.compute_dtype)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tf.Tensor:  # [B,H,W,C] → [B,H,W,n_out]
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

        self.fc0 = tf.keras.layers.Dense(width, dtype=self.compute_dtype)
        self.convs = [SpectralConv2D(width, width, modes1, modes2) for _ in range(4)]
        self.ws = [Conv1() for _ in range(4)]
        self.fc1 = tf.keras.layers.Dense(128, dtype=self.compute_dtype)
        self.fc2 = tf.keras.layers.Dense(n_out, dtype=self.compute_dtype)

    def _grid(self, x: tf.Tensor) -> tf.Tensor:  # [B,H,W,C] → [B,H,W,2]
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        gx = tf.tile(
            tf.reshape(tf.cast(tf.linspace(0.0, 1.0, H), x.dtype), [1, H, 1, 1]),
            [B, 1, W, 1],
        )
        gy = tf.tile(
            tf.reshape(tf.cast(tf.linspace(0.0, 1.0, W), x.dtype), [1, 1, W, 1]),
            [B, H, 1, 1],
        )
        return tf.concat([gx, gy], axis=-1)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tf.Tensor:  # [B,H,W,C] → [B,H,W,n_out]
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
            tf.keras.layers.Dense(n_filters, activation="gelu", dtype=self.compute_dtype)
            for _ in range(n_layers)
        ]
        self.out = tf.keras.layers.Dense(n_out, dtype=self.compute_dtype)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tf.Tensor:  # [B,H,W,C] → [B,H,W,n_out]
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)


class DahuNet(tf.keras.Model):
    """
    Unified physics-informed ice-flow emulator.

    Parameters
    ----------
    features : sequence of str
        Keys from FEATURES.  Fixed at construction — zero runtime
        cost (the loop is unrolled by tf.function at first trace).
    backend : {"cnn", "fno", "mlp"}
        Spatial backbone.  Hyperparameters are read from the same cfg paths
        as the original single-architecture models.
    """

    def __init__(
        self,
        cfg=None,
        nb_inputs=None,
        nb_outputs=None,
        *,
        input_names: list[str] | None = None,
        Nz: int | None = None,
        network_params: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Dual calling convention. See SIADecompNet for rationale.
        if cfg is not None:
            from .utils import parse_cfg_input_names_Nz, parse_cfg_network_params_strict
            input_names, Nz = parse_cfg_input_names_Nz(cfg, nb_inputs, nb_outputs)
            network_params = parse_cfg_network_params_strict(cfg, "DahuNet")

        if input_names is None or Nz is None or network_params is None:
            raise ValueError(
                "DahuNet: must provide either (cfg, nb_inputs, nb_outputs) or "
                "(input_names, Nz, network_params)."
            )

        # Reconstruction inputs
        self.input_names = list(input_names)
        self.Nz = int(Nz)
        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # Channel indices
        self.idx_thk = self.input_names.index("thk")
        self.idx_usurf = self.input_names.index("usurf")
        if "dX" not in self.input_names:
            raise ValueError(
                "DahuNet requires 'dX' in input_names "
                "(used by central-difference surface gradients)."
            )
        self.idx_dX = self.input_names.index("dX")

        # Settable from outside (pretraining attaches FixedChannelStandardization).
        self.input_normalizer = None

        # ------------------------------------------------------------------
        # Parse network_params
        # ------------------------------------------------------------------
        params = dict(network_params)

        common_keys = {"backend", "features"}
        cnn_keys = {"nb_out_filter", "nb_layers", "conv_ker_size", "residual"}
        fno_keys = {"width", "modes1", "modes2", "padding", "use_grid"}
        mlp_keys = {"nb_out_filter", "nb_layers"}

        self.backend_name = str(params.get("backend", "cnn"))
        features = tuple(params.get("features", FEATURES_DEFAULT))

        unknown_features = [f for f in features if f not in FEATURES]
        if unknown_features:
            raise ValueError(
                f"Unknown features: {unknown_features}\nAvailable: {sorted(FEATURES)}"
            )
        self.features = [str(f) for f in features]
        self.feature_fns = [FEATURES[f] for f in self.features]

        n_combined = self.nb_inputs + len(self.features)

        if self.backend_name == "cnn":
            allowed = common_keys | cnn_keys
            unexpected = sorted(set(params.keys()) - allowed)
            if unexpected:
                raise ValueError(
                    f"Unexpected keys in network_params for backend='cnn': {unexpected}. "
                    f"Allowed: {sorted(allowed)}"
                )
            self.backend_params = {
                "nb_out_filter": int(params.get("nb_out_filter", 32)),
                "nb_layers": int(params.get("nb_layers", 16)),
                "conv_ker_size": int(params.get("conv_ker_size", 3)),
                "residual": bool(params.get("residual", True)),
            }
            self._backend = _CNNBackend(
                n_combined,
                self.nb_outputs,
                n_filters=self.backend_params["nb_out_filter"],
                n_layers=self.backend_params["nb_layers"],
                ker_size=self.backend_params["conv_ker_size"],
                use_residual=self.backend_params["residual"],
            )
            self._dummy_H, self._dummy_W = 8, 8

        elif self.backend_name == "fno":
            allowed = common_keys | fno_keys
            unexpected = sorted(set(params.keys()) - allowed)
            if unexpected:
                raise ValueError(
                    f"Unexpected keys in network_params for backend='fno': {unexpected}. "
                    f"Allowed: {sorted(allowed)}"
                )
            modes1 = int(params.get("modes1", 8))
            modes2 = int(params.get("modes2", modes1))
            self.backend_params = {
                "width": int(params.get("width", 32)),
                "modes1": modes1,
                "modes2": modes2,
                "padding": int(params.get("padding", 9)),
                "use_grid": bool(params.get("use_grid", True)),
            }
            self._backend = _FNOBackend(
                n_combined,
                self.nb_outputs,
                width=self.backend_params["width"],
                modes1=modes1,
                modes2=modes2,
                padding=self.backend_params["padding"],
                use_grid=self.backend_params["use_grid"],
            )
            self._dummy_H = max(16, modes1 + 1)
            self._dummy_W = max(16, 2 * modes2 + 2)

        elif self.backend_name == "mlp":
            allowed = common_keys | mlp_keys
            unexpected = sorted(set(params.keys()) - allowed)
            if unexpected:
                raise ValueError(
                    f"Unexpected keys in network_params for backend='mlp': {unexpected}. "
                    f"Allowed: {sorted(allowed)}"
                )
            self.backend_params = {
                "nb_out_filter": int(params.get("nb_out_filter", 64)),
                "nb_layers": int(params.get("nb_layers", 8)),
            }
            self._backend = _MLPBackend(
                n_combined,
                self.nb_outputs,
                n_filters=self.backend_params["nb_out_filter"],
                n_layers=self.backend_params["nb_layers"],
            )
            self._dummy_H, self._dummy_W = 8, 8

        else:
            raise ValueError(
                f"Unknown backend '{self.backend_name}'. Choose: cnn, fno, mlp."
            )

        # Canonical reconstruction payload (native Python types only — the
        # manifest is yaml.safe_dump'd, so OmegaConf containers are rejected).
        self.network_params = {
            "backend": str(self.backend_name),
            "features": [str(f) for f in self.features],
            **{k: v for k, v in self.backend_params.items()},
        }

    # ----------------------------------------------------------------------
    # Pretraining / manifest contract
    # ----------------------------------------------------------------------
    def set_input_normalizer(self, layer: tf.keras.layers.Layer) -> None:
        self.input_normalizer = layer

    def resolved_params(self) -> Dict[str, Any]:
        # Rebuild every container fresh: Keras wraps lists/dicts assigned
        # to a Model in TrackedList / TrackedDict, which yaml.safe_dump
        # does not know how to represent.
        network_params = {
            "backend": str(self.backend_name),
            "features": [str(f) for f in self.features],
            **{str(k): v for k, v in self.backend_params.items()},
        }
        return {
            "input_names": [str(n) for n in self.input_names],
            "Nz": int(self.Nz),
            "network_params": network_params,
        }

    def build(self, input_shape) -> None:
        if self.built:
            return
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"DahuNet expects input_shape rank 4 [B, H, W, C], got {input_shape}"
            )
        channel_dim = input_shape[-1]
        channel_dim = self.nb_inputs if channel_dim is None else int(channel_dim)
        if channel_dim != self.nb_inputs:
            raise ValueError(
                f"Input channel mismatch: model expects {self.nb_inputs} channels "
                f"from input_names={self.input_names}, but build got C={channel_dim}."
            )
        H = self._dummy_H if input_shape[1] is None else max(self._dummy_H, int(input_shape[1]))
        W = self._dummy_W if input_shape[2] is None else max(self._dummy_W, int(input_shape[2]))
        dummy = tf.zeros((1, H, W, self.nb_inputs), dtype=self.compute_dtype)
        _ = self.call(dummy, training=False)
        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> tf.Tensor:  # [B,H,W,C_in] → [B,H,W,C_out]
        raw = tf.cast(inputs, self.compute_dtype)
        x = raw

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        proxies = _dict_proxies(raw, self.idx_thk, self.idx_usurf, self.idx_dX)
        features = tf.concat([fn(proxies) for fn in self.feature_fns], axis=-1)

        return self._backend(tf.concat([features, x], axis=-1), training=training)