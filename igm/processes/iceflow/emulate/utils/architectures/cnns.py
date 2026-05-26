from __future__ import annotations

import numpy as np
import tensorflow as tf
import math

from igm.utils.math.precision import normalize_precision
from .utils import (
    PeriodicBCAnsatz,
    PeriodicBCEnforcement,
    PeriodicBCFourier,
    PeriodicBCLayer,
    DTypeActivation,
)
from typing import Any, Dict


class CNN(tf.keras.Model):
    """
    Convolutional neural network with optional skip connection.

    Constructor:
        CNN(input_names=[...], Nz=..., network_params={...})

    network_params is a flat dict of architecture hyperparameters. All keys
    are optional; missing keys fall back to ``_DEFAULTS`` below. Unknown
    keys raise.

    Output convention:
        output[..., :Nz] = U_x(z)
        output[..., Nz:] = U_y(z)

    Therefore nb_outputs is fixed to 2 * Nz.
    """

    # Each entry is (default, coercer). Keys match conf/processes/iceflow.yaml
    # and become instance attribute names.
    _DEFAULTS: Dict[str, tuple] = {
        "nb_layers":             (16, int),
        "nb_out_filter":         (32, int),
        "conv_ker_size":         (3,  int),
        "activation":            ("LeakyReLU",      str),
        "weight_initialization": ("glorot_uniform", str),
        "batch_norm":            (False, bool),
        "residual":              (True,  bool),
        "separable":             (False, bool),
        "dropout_rate":          (0.0,   float),
        "l2_reg":                (0.0,   float),
        "cnn3d_for_vertical":    (False, bool),
        # Not exposed as top-level fields in iceflow.yaml; kept at code defaults.
        "precision":             ("float32", str),
        "use_skip":              (True,  bool),
        "leakyrelu_alpha":       (0.01,  float),
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

        # External normalizer. Attached by the emulator/artifact path,
        # not serialized as part of the architecture params.
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

        if self.nb_layers <= 0:
            raise ValueError(f"nb_layers must be > 0, got {self.nb_layers}")
        if self.nb_out_filter <= 0:
            raise ValueError(f"nb_out_filter must be > 0, got {self.nb_out_filter}")
        if self.conv_ker_size <= 0:
            raise ValueError(f"conv_ker_size must be > 0, got {self.conv_ker_size}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must satisfy 0 <= dropout_rate < 1, "
                f"got {self.dropout_rate}"
            )
        if self.cnn3d_for_vertical and self.Nz < 2:
            raise ValueError("cnn3d_for_vertical=True requires Nz >= 2")

        self.dtype_model = normalize_precision(self.precision)
        self.kernel_regularizer = (
            tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0.0 else None
        )

        self._build_layers()

    # ----------------------------------------------------------------------
    # Layer construction
    # ----------------------------------------------------------------------
    def _build_layers(self) -> None:
        """Create all Keras layer objects."""

        # Skip connection projection, input -> hidden feature width.
        if self.use_skip:
            self.skip_proj = tf.keras.layers.Conv2D(
                filters=self.nb_out_filter,
                kernel_size=(1, 1),
                padding="same",
                kernel_initializer=self.weight_initialization,
                kernel_regularizer=self.kernel_regularizer,
                dtype=self.dtype_model,
                name="skip_projection",
            )
        else:
            self.skip_proj = None

        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layers = []
        self.dropout_layers = []

        for i in range(self.nb_layers):
            if self.separable:
                conv = tf.keras.layers.SeparableConv2D(
                    filters=self.nb_out_filter,
                    kernel_size=(self.conv_ker_size, self.conv_ker_size),
                    padding="same",
                    depthwise_initializer=self.weight_initialization,
                    pointwise_initializer=self.weight_initialization,
                    depthwise_regularizer=self.kernel_regularizer,
                    pointwise_regularizer=self.kernel_regularizer,
                    dtype=self.dtype_model,
                    name=f"separable_conv_{i}",
                )
            else:
                conv = tf.keras.layers.Conv2D(
                    filters=self.nb_out_filter,
                    kernel_size=(self.conv_ker_size, self.conv_ker_size),
                    padding="same",
                    kernel_initializer=self.weight_initialization,
                    kernel_regularizer=self.kernel_regularizer,
                    dtype=self.dtype_model,
                    name=f"conv_{i}",
                )
            self.conv_layers.append(conv)

            if self.batch_norm:
                bn = tf.keras.layers.BatchNormalization(
                    dtype=self.dtype_model,
                    name=f"batch_norm_{i}",
                )
            else:
                bn = None
            self.batch_norm_layers.append(bn)

            self.activation_layers.append(self._make_activation(i))

            if self.dropout_rate > 0.0:
                dropout = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    dtype=self.dtype_model,
                    name=f"dropout_{i}",
                )
            else:
                dropout = None
            self.dropout_layers.append(dropout)

        # Optional 3D vertical extension.
        #
        # This intentionally differs from the old code's suspicious reshape:
        # the old implementation took only x[..., 0] and x[..., 1] after Conv3D.
        # Here the vertical-feature volume is flattened back into 2D channels
        # before the final 1x1 output layer.
        self.conv3d_layers = []
        self.upsample3d_layers = []

        if self.cnn3d_for_vertical:
            n_3d_layers = int(math.ceil(math.log2(self.Nz)))
            for i in range(n_3d_layers):
                filters_i = max(self.nb_out_filter // (2 ** (i + 1)), 1)

                conv3d = tf.keras.layers.Conv3D(
                    filters=filters_i,
                    kernel_size=(
                        self.conv_ker_size,
                        self.conv_ker_size,
                        self.conv_ker_size,
                    ),
                    padding="same",
                    kernel_initializer=self.weight_initialization,
                    kernel_regularizer=self.kernel_regularizer,
                    dtype=self.dtype_model,
                    name=f"conv3d_{i}",
                )
                upsample = tf.keras.layers.UpSampling3D(
                    size=(2, 1, 1),
                    dtype=self.dtype_model,
                    name=f"upsample3d_{i}",
                )

                self.conv3d_layers.append(conv3d)
                self.upsample3d_layers.append(upsample)

        self.output_layer = tf.keras.layers.Conv2D(
            filters=self.nb_outputs,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=self.weight_initialization,
            kernel_regularizer=self.kernel_regularizer,
            activation=None,
            dtype=self.dtype_model,
            name="output",
        )

    def _make_activation(self, i: int) -> tf.keras.layers.Layer:
        activation_name = self.activation.lower()

        if activation_name == "leakyrelu":
            return tf.keras.layers.LeakyReLU(
                dtype=self.dtype_model,
                name=f"leakyrelu_{i}",
            )

        if activation_name in ("swish", "silu"):
            return tf.keras.layers.Activation(
                tf.nn.swish,
                dtype=self.dtype_model,
                name=f"swish_{i}",
            )

        if activation_name == "gelu":
            return tf.keras.layers.Activation(
                tf.nn.gelu,
                dtype=self.dtype_model,
                name=f"gelu_{i}",
            )

        return tf.keras.layers.Activation(
            self.activation,
            dtype=self.dtype_model,
            name=f"{self.activation}_{i}",
        )

    # ----------------------------------------------------------------------
    # Minimal reconstruction manifest payload
    # ----------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        """
        Return exactly the minimal constructor payload needed to rebuild the
        model structure before attaching weights / external normalizer.
        """
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": {k: getattr(self, k) for k in self._DEFAULTS},
        }

    # ----------------------------------------------------------------------
    # Keras build
    # ----------------------------------------------------------------------
    def build(self, input_shape) -> None:
        """
        Explicit deterministic build for subclassed-model compatibility.

        This creates weights in forward-pass order via a dummy call, mirroring
        the SIADecompNet pattern.
        """
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"CNN expects input_shape rank 4 [B, H, W, C], got {input_shape}"
            )

        channel_dim = input_shape[-1]
        if channel_dim is None:
            channel_dim = self.nb_inputs
        else:
            channel_dim = int(channel_dim)

        if channel_dim != self.nb_inputs:
            raise ValueError(
                f"Input channel mismatch: model expects {self.nb_inputs} channels "
                f"from input_names={self.input_names}, but build got C={channel_dim}."
            )

        batch_dim = 1 if input_shape[0] is None else int(input_shape[0])
        height_dim = 4 if input_shape[1] is None else int(input_shape[1])
        width_dim = 4 if input_shape[2] is None else int(input_shape[2])

        dummy = tf.zeros(
            shape=(batch_dim, height_dim, width_dim, channel_dim),
            dtype=self.dtype_model,
        )

        _ = self.call(dummy, training=False)
        super().build(input_shape)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def call(self, inputs, training=None):
        x = tf.cast(inputs, self.dtype_model)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
            x = tf.cast(x, self.dtype_model)

        if self.use_skip:
            skip = self.skip_proj(x)
        else:
            skip = None

        for i in range(self.nb_layers):
            residual_in = x

            x = self.conv_layers[i](x)

            bn = self.batch_norm_layers[i]
            if bn is not None:
                x = bn(x, training=training)

            x = self.activation_layers[i](x)

            dropout = self.dropout_layers[i]
            if dropout is not None:
                x = dropout(x, training=training)

            if (
                self.residual
                and i % 2 == 1
                and residual_in.shape[-1] is not None
                and x.shape[-1] == residual_in.shape[-1]
            ):
                x = x + residual_in

        if skip is not None:
            x = x + skip

        if self.cnn3d_for_vertical:
            x = self._vertical_extension_3d(x)

        return self.output_layer(x)

    def _vertical_extension_3d(self, x: tf.Tensor) -> tf.Tensor:
        """
        Optional vertical extension path.

        Input:
            x: [B, H, W, C]

        Internal:
            [B, 1, H, W, C] -> repeated/conv3d upsampling in vertical dimension

        Output:
            [B, H, W, Nz * C3d]

        The final 2D output_layer maps this flattened vertical-feature stack
        to [B, H, W, 2*Nz].
        """
        x = tf.expand_dims(x, axis=1)  # [B, Z=1, H, W, C]

        for conv3d, upsample in zip(self.conv3d_layers, self.upsample3d_layers):
            x = conv3d(x)
            x = upsample(x)

        # If Nz is not an exact power of two, crop the vertical dimension.
        x = x[:, : self.Nz, :, :, :]  # [B, Nz, H, W, C3d]

        # [B, Nz, H, W, C3d] -> [B, H, W, Nz, C3d]
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])

        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3] * shape[4]

        return tf.reshape(x, [batch, height, width, channels])

    # ----------------------------------------------------------------------
    # Keras serialization compatibility
    # ----------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        config.update(self.resolved_params())
        return config


class CNNPatch(tf.keras.Model):
    """
    Simple convolutional neural network with optional skip connection.
    """

    def __init__(
        self, cfg, nb_inputs, nb_outputs, input_normalizer=None, use_skip=True
    ):  # New parameter
        super(CNNPatch, self).__init__()
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)
        self.input_normalizer = input_normalizer
        self.use_skip = use_skip  # Store flag

        # Build convolutional layers
        self.conv_layers = []
        self.activations = []
        n_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        n_filters = cfg.processes.iceflow.emulator.network.nb_out_filter

        for i in range(n_layers):
            layer = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                padding="same",
                dtype=self.dtype_model,
            )
            activation = DTypeActivation(
                activation_name=cfg.processes.iceflow.emulator.network.activation,
                name=f"{cfg.processes.iceflow.emulator.network.activation}_{i}",
                dtype=self.dtype_model,
            )
            self.conv_layers.append(layer)
            self.activations.append(activation)

        # Projection layer for skip connection (only create if using skip)
        if self.use_skip:
            self.skip_proj = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(1, 1),
                padding="same",
                dtype=self.dtype_model,
            )
        else:
            self.skip_proj = None

        # Output layer
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            activation=None,
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Store skip connection if enabled
        if self.use_skip:
            skip = self.skip_proj(x)

        # Main path through convolutional layers
        for conv, activation in zip(self.conv_layers, self.activations):
            x = conv(x)
            x = activation(x)

        # Add skip connection if enabled
        if self.use_skip:
            x = x + skip

        # Output layer
        outputs = self.output_layer(x)

        return outputs


class CNNPeriodic(tf.keras.Model):
    """
    Simple convolutional neural network with skip connection.
    Optional periodic boundary condition enforcement via Fourier features.
    """

    def __init__(
        self,
        cfg,
        nb_inputs,
        nb_outputs,
        use_periodic_bc=True,
        num_frequencies=3,
        periodic_enforcement="fourier",
    ):  # New parameter
        super(CNNPeriodic, self).__init__()
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)

        # Normalization layer (set later on)
        self.input_normalizer = None

        # Periodic BC layer for INPUT (Fourier features)
        self.use_periodic_bc = use_periodic_bc
        if use_periodic_bc:
            self.periodic_layer = PeriodicBCLayer(
                num_frequencies=num_frequencies,
                name="periodic_bc_input",
                dtype=self.dtype_model,
            )
            num_fourier_features = 4 * (num_frequencies**2 - 1)

            # Periodic BC enforcement for OUTPUT
            if periodic_enforcement == "hard":
                self.periodic_output = PeriodicBCEnforcement(name="periodic_bc_output")
            elif periodic_enforcement == "ansatz":
                self.periodic_output = PeriodicBCAnsatz(name="periodic_bc_output")
            elif periodic_enforcement == "fourier":
                self.periodic_output = PeriodicBCFourier(
                    name="periodic_bc_output", dtype=self.dtype_model
                )
            else:
                self.periodic_output = None
        else:
            self.periodic_layer = None
            self.periodic_output = None

        # Build convolutional layers
        self.conv_layers = []
        self.activations = []
        n_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        n_filters = cfg.processes.iceflow.emulator.network.nb_out_filter

        for i in range(n_layers):
            layer = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                padding="same",
                dtype=self.dtype_model,
            )

            # Custom activation layer that uses TF function as keras was overriding datatype
            # Might be achievable with tf.keras.backend float64 or something
            activation = DTypeActivation(
                activation_name=cfg.processes.iceflow.emulator.network.activation,
                name=f"{cfg.processes.iceflow.emulator.network.activation}_{i}",
                dtype=self.dtype_model,
            )
            # activation.compute_dtype = self.dtype_model
            # print(activation.compute_dtype)

            self.conv_layers.append(layer)
            self.activations.append(activation)

        # Projection layer for skip connection (1×1 conv)
        self.skip_proj = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            padding="same",
            dtype=self.dtype_model,
        )

        # Output layer
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            activation=None,
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        # Apply normalization if specified
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Apply periodic BC layer for INPUT (Fourier features)
        if self.use_periodic_bc and self.periodic_layer is not None:
            x = self.periodic_layer(x)

        # Store skip connection
        skip = self.skip_proj(x)

        # Main path through convolutional layers
        for conv, activation in zip(self.conv_layers, self.activations):
            x = conv(x)
            x = activation(x)

        # Add skip connection
        x = x + skip

        # Output layer
        outputs = self.output_layer(x)

        # Apply periodic BC enforcement for OUTPUT
        if self.use_periodic_bc and self.periodic_output is not None:
            outputs = self.periodic_output(outputs)

        return outputs


class CNNSkip(tf.keras.Model):
    """
    Simple convolutional neural network with optional skip connection.
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, use_skip=True):  # New parameter
        super(CNNSkip, self).__init__()
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)
        self.input_normalizer = None
        self.use_skip = use_skip  # Store flag

        # Build convolutional layers
        self.conv_layers = []
        self.activations = []
        n_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        n_filters = cfg.processes.iceflow.emulator.network.nb_out_filter

        for _ in range(n_layers):
            layer = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                padding="same",
                dtype=self.dtype_model,
            )
            activation = DTypeActivation(
                activation_name=cfg.processes.iceflow.emulator.network.activation,
                name=f"{cfg.processes.iceflow.emulator.network.activation}_{_}",
                dtype=self.dtype_model,
            )
            self.conv_layers.append(layer)
            self.activations.append(activation)

        # Projection layer for skip connection (only create if using skip)
        if self.use_skip:
            self.skip_proj = tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(1, 1),
                padding="same",
                dtype=self.dtype_model,
            )
        else:
            self.skip_proj = None

        # Output layer
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            activation=None,
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Store skip connection if enabled
        if self.use_skip:
            skip = self.skip_proj(x)

        # Main path through convolutional layers
        for conv, activation in zip(self.conv_layers, self.activations):
            x = conv(x)
            x = activation(x)

        # Add skip connection if enabled
        if self.use_skip:
            x = x + skip

        # Output layer
        outputs = self.output_layer(x)

        return outputs