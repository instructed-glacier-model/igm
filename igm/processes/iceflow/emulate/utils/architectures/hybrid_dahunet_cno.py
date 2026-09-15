"""CNN-backed DahuNet plus CNO, replacing the hybrid's spectral branch."""

from __future__ import annotations

import math
from typing import Any

import tensorflow as tf

from .cno import CNO
from .dahunet import DahuNet, FEATURES_DEFAULT


class HybridDahuNetCNO(CNO):
    """Sum of local physics-informed CNN and multiscale CNO velocities.

    CNO parameters follow :class:`CNO`; the four CNN parameters match the
    existing ``hybrid_dahunet_fno`` architecture. The two branches share the
    external input normalizer, but have independent trainable weights.
    ``operator_scale`` optionally conditions only the CNO velocity branch;
    the default value one preserves the unscaled branch sum.
    """

    _DEFAULTS = {
        **CNO._DEFAULTS,
        "features": (FEATURES_DEFAULT, tuple),
        "nb_out_filter": (24, int),
        "nb_layers": (6, int),
        "conv_ker_size": (3, int),
        "residual": (True, bool),
        "operator_scale": (1.0, float),
    }

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_names=input_names, Nz=Nz, network_params=network_params, **kwargs
        )
        if not math.isfinite(self.operator_scale) or self.operator_scale <= 0:
            raise ValueError("operator_scale must be positive")
        self.dahunet_branch = DahuNet(
            input_names=self.input_names,
            Nz=self.Nz,
            network_params={
                "backend": "cnn",
                "features": self.features or FEATURES_DEFAULT,
                "nb_out_filter": self.nb_out_filter,
                "nb_layers": self.nb_layers,
                "conv_ker_size": self.conv_ker_size,
                "residual": self.residual,
            },
            name="dahunet_branch",
            dtype=self.dtype_policy,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.dahunet_branch.input_normalizer is not self.input_normalizer:
            self.dahunet_branch.input_normalizer = self.input_normalizer
        return self.operator_scale * super().call(
            inputs, training=training
        ) + self.dahunet_branch(inputs, training=training)
