"""Hybrid DahuNet + FNO architecture.

The model evaluates two independently trainable branches on the same inputs:

* :class:`DahuNet`, fixed to its CNN backend, with its physics-derived
  features (the ``backend`` parameter is not exposed: a DahuNet with the FNO
  backend is a different model, select it as ``architecture: dahunet``);
* the standalone :class:`FNO` architecture.

Their velocity outputs are added elementwise without a learned or fixed
weighting. For ``Nz`` vertical levels, both branches and the sum have shape
``[batch, y, x, 2 * Nz]``.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .dahunet import DahuNet, FEATURES_DEFAULT
from .nos import FNO


class HybridDahuNetFNO(tf.keras.Model):
    """Sum of a CNN-backed DahuNet and a standalone FNO."""

    # Flat parameters deliberately follow the existing branch names so an
    # experiment configured for dahunet can switch architecture without
    # rewriting its CNN hyperparameters. FNO-only keys use their native names.
    _DEFAULTS: Dict[str, tuple[Any, type]] = {
        "features": (FEATURES_DEFAULT, tuple),
        "nb_out_filter": (32, int),
        "nb_layers": (16, int),
        "conv_ker_size": (3, int),
        "residual": (True, bool),
        "width": (32, int),
        "modes1": (8, int),
        "modes2": (8, int),
        "padding": (9, int),
        "use_grid": (True, bool),
        "projection_width": (128, int),
        "fno_layers": (4, int),
        "factorization_rank": (0, int),
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

        features = [str(name) for name in params.get("features", FEATURES_DEFAULT)]
        dahunet_params = {
            "backend": "cnn",
            "features": features,
            "nb_out_filter": int(params.get("nb_out_filter", 32)),
            "nb_layers": int(params.get("nb_layers", 16)),
            "conv_ker_size": int(params.get("conv_ker_size", 3)),
            "residual": bool(params.get("residual", True)),
        }
        fno_params = {
            "width": int(params.get("width", 32)),
            "modes1": int(params.get("modes1", 8)),
            "modes2": int(params.get("modes2", 8)),
            "padding": int(params.get("padding", 9)),
            "use_grid": bool(params.get("use_grid", True)),
            "projection_width": int(params.get("projection_width", 128)),
            "n_layers": int(params.get("fno_layers", 4)),
            "factorization_rank": int(params.get("factorization_rank", 0)),
        }

        self.dahunet_branch = DahuNet(
            input_names=self.input_names,
            Nz=self.Nz,
            network_params=dahunet_params,
            name="dahunet_branch",
        )
        self.fno_branch = FNO(
            input_names=self.input_names,
            Nz=self.Nz,
            network_params=fno_params,
            name="fno_branch",
        )

    def _share_input_normalizer(self) -> None:
        """Attach the composite's normalizer to both unbuilt branches."""
        for branch in (self.dahunet_branch, self.fno_branch):
            if branch.input_normalizer is not self.input_normalizer:
                branch.input_normalizer = self.input_normalizer

    def resolved_params(self) -> Dict[str, Any]:
        """Return the minimal constructor payload used by emulator artifacts."""
        dahunet = self.dahunet_branch.resolved_params()["network_params"]
        fno = self.fno_branch.resolved_params()["network_params"]
        return {
            "input_names": [str(name) for name in self.input_names],
            "Nz": int(self.Nz),
            "network_params": {
                "features": [str(name) for name in dahunet["features"]],
                "nb_out_filter": int(dahunet["nb_out_filter"]),
                "nb_layers": int(dahunet["nb_layers"]),
                "conv_ker_size": int(dahunet["conv_ker_size"]),
                "residual": bool(dahunet["residual"]),
                "width": int(fno["width"]),
                "modes1": int(fno["modes1"]),
                "modes2": int(fno["modes2"]),
                "padding": int(fno["padding"]),
                "use_grid": bool(fno["use_grid"]),
                "projection_width": int(fno["projection_width"]),
                "fno_layers": int(fno["n_layers"]),
                "factorization_rank": int(fno["factorization_rank"]),
            },
        }

    def build(self, input_shape) -> None:
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                "HybridDahuNetFNO expects input_shape rank 4 [B, H, W, C], "
                f"got {input_shape}"
            )
        channels = self.nb_inputs if input_shape[-1] is None else int(input_shape[-1])
        if channels != self.nb_inputs:
            raise ValueError(
                f"Input channel mismatch: model expects {self.nb_inputs} channels "
                f"from input_names={self.input_names}, but build got C={channels}."
            )

        height = self.fno_branch._dummy_H
        width = self.fno_branch._dummy_W
        if input_shape[1] is not None:
            height = max(height, int(input_shape[1]))
        if input_shape[2] is not None:
            width = max(width, int(input_shape[2]))

        self._share_input_normalizer()
        dummy = tf.zeros((1, height, width, channels), dtype=self.compute_dtype)
        _ = self.call(dummy, training=False)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        self._share_input_normalizer()
        dahunet_output = self.dahunet_branch(inputs, training=training)
        fno_output = self.fno_branch(inputs, training=training)
        return dahunet_output + fno_output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.resolved_params())
        return config
