import tensorflow as tf
import numpy as np

from igm.utils.math.precision import normalize_precision


class MLP(tf.keras.Model):
    """
    Simple multi-layer perceptron (fully connected network)
    """

    def __init__(self, cfg, nb_inputs, nb_outputs):
        super(MLP, self).__init__()

        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)

        self.input_normalizer = None

        # Build hidden layers
        self.hidden_layers = []

        for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
            layer = tf.keras.layers.Dense(
                units=cfg.processes.iceflow.emulator.network.nb_out_filter,
                activation=cfg.processes.iceflow.emulator.network.activation,
                dtype=self.dtype_model,
            )
            self.hidden_layers.append(layer)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            units=nb_outputs,
            activation=None,
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Output layer
        outputs = self.output_layer(x)

        return outputs


class FourierMLP(tf.keras.Model):
    """
    MLP with proper coordinate handling for spatial problems
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(FourierMLP, self).__init__()

        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)

        self.input_normalizer = input_normalizer
        self.nb_inputs = nb_inputs

        nb_filters = cfg.processes.iceflow.emulator.network.nb_out_filter
        nb_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)
        activation = cfg.processes.iceflow.emulator.network.activation

        # Fourier features (crucial for spatial learning)
        self.fourier_scale = 1.0  # Adjust this if needed
        self.fourier_dim = 64

        # +2 for normalized (x,y) coordinates
        coord_input_dim = nb_inputs + 2

        self.B = tf.Variable(
            tf.random.normal(
                [coord_input_dim, self.fourier_dim], dtype=self.dtype_model
            ) * self.fourier_scale,
            trainable=False,
            dtype=self.dtype_model,
        )

        # MLP layers
        self.dense_layers = []
        input_dim = self.fourier_dim * 2

        # First layer handles Fourier features
        self.dense_layers.append(
            tf.keras.layers.Dense(
                nb_filters,
                activation=activation,
                kernel_initializer="glorot_uniform",
                dtype=self.dtype_model,
            )
        )

        # Hidden layers with residual connections
        for i in range(nb_layers - 1):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    nb_filters,
                    activation=activation,
                    kernel_initializer="glorot_uniform",
                    dtype=self.dtype_model,
                )
            )

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            nb_outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            dtype=self.dtype_model,
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        # Create normalized coordinate grid
        # CRITICAL: Normalize to [-1, 1] for both dimensions
        y_coords = tf.linspace(-1.0, 1.0, height)
        x_coords = tf.linspace(-1.0, 1.0, width)

        xx, yy = tf.meshgrid(x_coords, y_coords)

        # Expand to match batch size
        xx = tf.cast(xx, self.dtype_model)
        yy = tf.cast(yy, self.dtype_model)
        xx = tf.tile(xx[None, :, :, None], [batch_size, 1, 1, 1])
        yy = tf.tile(yy[None, :, :, None], [batch_size, 1, 1, 1])

        # Concatenate coordinates with input features
        x = tf.concat([x, xx, yy], axis=-1)

        # Fourier features
        x_proj = tf.matmul(x, self.B)
        x = tf.concat([tf.sin(2 * np.pi * x_proj), tf.cos(2 * np.pi * x_proj)], axis=-1)

        # MLP with residual connections
        for i, dense in enumerate(self.dense_layers):
            if i == 0:
                x = dense(x)
            else:
                residual = x
                x = dense(x) + residual

        outputs = self.output_layer(x)

        return outputs
