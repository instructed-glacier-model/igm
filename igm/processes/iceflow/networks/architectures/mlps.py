from __future__ import annotations

from typing import Any, Dict
import tensorflow as tf

from igm.utils.math.precision import normalize_precision


class MLP(tf.keras.Model):
    """
    Simple multi-layer perceptron (fully connected network).

    Constructor:
        MLP(input_names=[...], Nz=..., network_params={...})

    network_params is a flat dict of architecture hyperparameters. All keys
    are optional; missing keys fall back to ``_DEFAULTS`` below.
    """

    _DEFAULTS: Dict[str, tuple] = {
        "nb_layers":     (4,         int),
        "nb_out_filter": (64,        int),
        "activation":    ("gelu",    str),
        "precision":     ("float32", str),
    }

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_names = [str(x) for x in input_names]
        self.Nz = int(Nz)
        if self.Nz <= 0:
            raise ValueError(f"Nz must be > 0, got {self.Nz}")

        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz
        self.input_normalizer = None

        params = dict(network_params) if network_params else {}
        unexpected = sorted(set(params) - set(self._DEFAULTS))
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys: {sorted(self._DEFAULTS)}"
            )
        for k, (default, cast) in self._DEFAULTS.items():
            setattr(self, k, cast(params.get(k, default)))

        self.dtype_model = normalize_precision(self.precision)

        self.hidden_layers = [
            tf.keras.layers.Dense(
                self.nb_out_filter,
                activation=self.activation,
                dtype=self.dtype_model,
                name=f"dense_{i}",
            )
            for i in range(self.nb_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(
            self.nb_outputs,
            activation=None,
            dtype=self.dtype_model,
            name="output",
        )

    def build(self, input_shape) -> None:
        if self.built:
            return
        input_shape = tf.TensorShape(input_shape)
        channel_dim = input_shape[-1]
        channel_dim = self.nb_inputs if channel_dim is None else int(channel_dim)
        dummy = tf.zeros(shape=(1, 4, 4, channel_dim), dtype=self.dtype_model)
        _ = self.call(dummy, training=False)
        super().build(input_shape)

    def resolved_params(self) -> Dict[str, Any]:
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": {k: getattr(self, k) for k in self._DEFAULTS},
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.resolved_params())
        return config

    def call(self, inputs, training=None):
        x = tf.cast(inputs, self.dtype_model)
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
            x = tf.cast(x, self.dtype_model)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
