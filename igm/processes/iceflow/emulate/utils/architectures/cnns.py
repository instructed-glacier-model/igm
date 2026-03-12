import numpy as np
import tensorflow as tf

from igm.utils.math.precision import normalize_precision
from .utils import (
    PeriodicBCAnsatz,
    PeriodicBCEnforcement,
    PeriodicBCFourier,
    PeriodicBCLayer,
    DTypeActivation,
)


class CNN(tf.keras.Model):
    """
    Convolutional neural network with optional skip connection.

    !!! Original CNN (not fully compared to existing CNN but should be close.)

    Features:
    - Skip connection from input to pre-output
    - Optional batch normalization
    - Optional residual connections (every 2 layers)
    - Optional separable convolutions
    - Optional dropout
    - Optional L2 regularization
    - Support for LeakyReLU and other activations
    - Optional 3D convolution for vertical extension

    Args:
        cfg: Configuration object containing network parameters
        nb_inputs: Number of input channels
        nb_outputs: Number of output channels
        input_normalizer: Optional normalization layer to apply to inputs
        use_skip: Whether to use skip connection from input to pre-output (default: True)
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, use_skip=True):
        super(CNN, self).__init__()

        # Store configuration
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)
        self.use_skip = use_skip
        self.input_normalizer = None

        # Extract network configuration
        net_cfg = cfg.processes.iceflow.emulator.network
        self.n_layers = int(net_cfg.nb_layers)
        self.n_filters = net_cfg.nb_out_filter
        self.kernel_size = net_cfg.conv_ker_size
        self.activation_name = net_cfg.activation
        self.weight_init = net_cfg.weight_initialization

        # Optional features
        self.use_batch_norm = hasattr(net_cfg, "batch_norm") and net_cfg.batch_norm
        self.use_residual = hasattr(net_cfg, "residual") and net_cfg.residual
        self.use_separable = hasattr(net_cfg, "separable") and net_cfg.separable
        self.dropout_rate = (
            net_cfg.dropout_rate if hasattr(net_cfg, "dropout_rate") else 0.0
        )

        # L2 regularization
        if hasattr(net_cfg, "l2_reg"):
            self.kernel_regularizer = tf.keras.regularizers.l2(net_cfg.l2_reg)
        else:
            self.kernel_regularizer = None

        # 3D convolution for vertical extension
        self.use_3d = (
            hasattr(net_cfg, "cnn3d_for_vertical") and net_cfg.cnn3d_for_vertical
        )
        if self.use_3d:
            self.n_vertical = cfg.processes.iceflow.numerics.Nz

        # Build layers
        self._build_layers(nb_inputs, nb_outputs)

        # Call build to initialize weights
        self.build(input_shape=[None, None, None, nb_inputs])

    def _build_layers(self, nb_inputs, nb_outputs):
        """Build all network layers."""

        # Skip connection projection (1x1 conv)
        if self.use_skip:
            self.skip_proj = tf.keras.layers.Conv2D(
                filters=self.n_filters,
                kernel_size=(1, 1),
                padding="same",
                kernel_initializer=self.weight_init,
                kernel_regularizer=self.kernel_regularizer,
                dtype=self.dtype_model,
                name="skip_projection",
            )

        # Main convolutional layers
        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layers = []
        self.dropout_layers = []

        for i in range(self.n_layers):
            # Convolutional layer
            if self.use_separable:
                conv = tf.keras.layers.SeparableConv2D(
                    filters=self.n_filters,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    depthwise_initializer=self.weight_init,
                    pointwise_initializer=self.weight_init,
                    padding="same",
                    depthwise_regularizer=self.kernel_regularizer,
                    pointwise_regularizer=self.kernel_regularizer,
                    dtype=self.dtype_model,
                    name=f"separable_conv_{i}",
                )
            else:
                conv = tf.keras.layers.Conv2D(
                    filters=self.n_filters,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    kernel_initializer=self.weight_init,
                    padding="same",
                    kernel_regularizer=self.kernel_regularizer,
                    dtype=self.dtype_model,
                    name=f"conv_{i}",
                )
            self.conv_layers.append(conv)

            # Batch normalization
            if self.use_batch_norm:
                bn = tf.keras.layers.BatchNormalization(
                    dtype=self.dtype_model, name=f"batch_norm_{i}"
                )
                self.batch_norm_layers.append(bn)
            else:
                self.batch_norm_layers.append(None)

            # Activation
            activation_name = self.activation_name.lower()
            if activation_name == "leakyrelu":
                activation = tf.keras.layers.LeakyReLU(
                    alpha=0.01,
                    dtype=self.dtype_model,      
                    name=f"leakyrelu_{i}",
                )
            elif activation_name in ("swish", "silu"):
                activation = tf.keras.layers.Activation(
                    tf.nn.swish,
                    dtype=self.dtype_model,       
                    name=f"swish_{i}",
                )
            elif activation_name == "gelu":
                activation = tf.keras.layers.Activation(
                    tf.nn.gelu,
                    dtype=self.dtype_model,           
                    name=f"gelu_{i}",
                )
            else:
                activation = tf.keras.layers.Activation(
                    self.activation_name,
                    dtype=self.dtype_model,          
                    name=f"{self.activation_name}_{i}",
                )
            self.activation_layers.append(activation)

            # Dropout
            if self.dropout_rate > 0:
                dropout = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    dtype=self.dtype_model,     
                    name=f"dropout_{i}",
                )
                self.dropout_layers.append(dropout)
            else:
                self.dropout_layers.append(None)

        # 3D convolution layers for vertical extension
        if self.use_3d:
            self.conv3d_layers = []
            self.upsample3d_layers = []

            n_3d_layers = int(np.log(self.n_vertical) / np.log(2))
            for i in range(n_3d_layers):
                conv3d = tf.keras.layers.Conv3D(
                    filters=int(self.n_filters / (2 ** (i + 1))),
                    kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                    padding="same",
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

        # Output layer (1x1 conv)
        self.output_layer = tf.keras.layers.Conv2D(
            filters=nb_outputs,
            kernel_size=(1, 1),
            kernel_initializer=self.weight_init,
            activation=None,
            dtype=self.dtype_model,
            name="output",
        )

    def call(self, inputs, training=None):
        """Forward pass through the network."""

        x = tf.cast(inputs, self.dtype_model) 

        # Apply input normalization if provided
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)
            x = tf.cast(x, self.dtype_model)

        # Store skip connection from normalized input
        if self.use_skip:
            skip = self.skip_proj(x)

        # Main convolutional path
        for i in range(self.n_layers):
            # Store for potential residual connection
            residual_in = x

            # Convolution
            x = self.conv_layers[i](x)

            # Batch normalization
            if self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)

            # Activation
            x = self.activation_layers[i](x)

            # Dropout
            if self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)

            # Residual connection (every 2 layers, if channel dimensions match)
            if (
                self.use_residual
                and i % 2 == 1
                and x.shape[-1] == residual_in.shape[-1]
            ):
                x = x + residual_in

        # Add skip connection from input to pre-output
        if self.use_skip:
            x = x + skip

        # 3D convolution for vertical extension
        if self.use_3d:
            # Add vertical dimension
            x = tf.expand_dims(x, axis=1)

            # Apply 3D convolutions with upsampling
            for conv3d, upsample in zip(self.conv3d_layers, self.upsample3d_layers):
                x = conv3d(x)
                x = upsample(x)

            # Reshape back to 2D
            # TO DO: CHECK IF THIS IS CORRECT
            # Why is it only taking the first two channels?
            x = tf.transpose(
                tf.concat([x[:, :, :, :, 0], x[:, :, :, :, 1]], axis=1),
                perm=[0, 2, 3, 1],
            )

        # Output layer
        outputs = self.output_layer(x)

        return outputs

    def get_config(self):
        """Return configuration for serialization."""
        config = super(CNN, self).get_config()
        config.update(
            {
                "use_skip": self.use_skip,
                "n_layers": self.n_layers,
                "n_filters": self.n_filters,
                "kernel_size": self.kernel_size,
                "activation_name": self.activation_name,
                "use_batch_norm": self.use_batch_norm,
                "use_residual": self.use_residual,
                "use_separable": self.use_separable,
                "dropout_rate": self.dropout_rate,
                "use_3d": self.use_3d,
            }
        )
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