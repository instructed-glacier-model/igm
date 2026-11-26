import tensorflow as tf
from typing import Tuple, Dict
import numpy as np


class FixedAffineLayer(tf.keras.layers.Layer):
    """
    Standardization layer with manually specified mean and variance for each channel.
    """

    def __init__(
        self, offsets: Dict, variances: Dict, epsilon=1e-6, dtype="float32", **kwargs
    ):
        """
        Args:
            offsets: Array of mean values for each channel
            variances: Array of variance values for each channel (optional, defaults to ones)
            epsilon: Small constant for numerical stability
            dtype: Data type for the layer
        """

        super(FixedAffineLayer, self).__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon

        offsets = offsets.values()
        variances = variances.values()

        # Default variances to ones if not provided
        if offsets is None:
            self.offsets = tf.convert_to_tensor(
                np.zeros_like(self.offsets, dtype=dtype)
            )
        else:
            self.offsets = np.fromiter(offsets, dtype=dtype)

        if variances is None:
            self.variances = tf.convert_to_tensor(
                np.ones_like(self.offsets, dtype=dtype)
            )
        else:
            self.variances = tf.convert_to_tensor(np.fromiter(variances, dtype=dtype))

        if self.offsets.ndim == 1:
            self.offsets = np.reshape(self.offsets, (1, 1, 1, -1))
        if self.variances.ndim == 1:
            self.variances = np.reshape(self.variances, (1, 1, 1, -1))

        # Validate shapes match
        if len(self.offsets) != len(self.variances):
            raise ValueError(
                f"offsets and variances must have same length, "
                f"got {len(self.offsets)} and {len(self.variances)}"
            )

    def build(self, input_shape):
        """Build the layer - validate input channels and create weight tensors."""
        nb_channels = input_shape[-1]

        # Validate that offsets match input channels
        if self.offsets.size != nb_channels:
            raise ValueError(
                f"Expected offsets of length {nb_channels}, got {self.offsets.size}"
            )
        if self.variances.size != nb_channels:
            raise ValueError(
                f"Expected offsets of length {nb_channels}, got {self.variances.size}"
            )

        # Create non-trainable weights for mean and variance
        # Shape: [1, 1, 1, nb_channels] for broadcasting over [batch, height, width, channels]
        self.mean = self.add_weight(
            name="mean",
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.offsets),
            trainable=False,
            dtype=self.dtype,
        )

        self.variance = self.add_weight(
            name="variance",
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.variances),
            trainable=False,
            dtype=self.dtype,
        )

        super(FixedAffineLayer, self).build(input_shape)

    def call(self, inputs):
        """Apply normalization: (x - mean) / std"""
        std = tf.sqrt(self.variance + self.epsilon)
        normalized = (inputs - self.mean) / std
        return normalized

    def set_stats(self, means, variances):
        pass  # fixed does not do anything

    def compute_stats(self, inputs):  # fix liskov violation here
        return 0, 0

    def get_config(self):
        """Enable serialization."""
        config = super(FixedAffineLayer, self).get_config()
        config.update(
            {
                "offsets": self.offsets.tolist(),
                "variances": self.variances.tolist(),
                "epsilon": self.epsilon,
            }
        )
        return config


class AdaptiveAffineLayer(tf.keras.layers.Layer):
    """
    Standardization layer with manually specified mean and variance for each channel.
    Can be built without data; mean and variance are initialized in build() based on input channels.
    Supports updating stats via set_stats().
    """

    def __init__(self, nb_channels=None, epsilon=1e-6, dtype="float32", **kwargs):
        super(AdaptiveAffineLayer, self).__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon
        self.mean = None
        self.variance = None
        self.nb_channels = nb_channels
        self.means_np = None
        self.variances_np = None

        # If nb_channels is provided, build immediately
        if nb_channels is not None:
            self.build((None, None, None, nb_channels))

    def build(self, input_shape):
        import numpy as np

        if self.built:
            return

        nb_channels = (
            self.nb_channels if self.nb_channels is not None else input_shape[-1]
        )
        self.nb_channels = nb_channels

        # Initialize mean=0, variance=1 if not already set
        if self.means_np is None:
            self.means_np = np.zeros((1, 1, 1, nb_channels), dtype=np.float32)
        if self.variances_np is None:
            self.variances_np = np.ones((1, 1, 1, nb_channels), dtype=np.float32)

        # Create TF variables
        self.mean = self.add_weight(
            name="mean",
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.means_np),
            trainable=False,
            dtype=self.dtype,
        )
        self.variance = self.add_weight(
            name="variance",
            shape=(1, 1, 1, nb_channels),
            initializer=tf.constant_initializer(self.variances_np),
            trainable=False,
            dtype=self.dtype,
        )
        super(AdaptiveAffineLayer, self).build(input_shape)

    def call(self, inputs):
        std = tf.sqrt(self.variance + self.epsilon)
        return (inputs - self.mean) / std

    def set_stats(self, means, variances):
        """
        Update the mean and variance without recreating the layer.
        Input can be shape [C] or [1,1,1,C] or a Tensor.
        """
        means = tf.convert_to_tensor(means, dtype=self.dtype)
        variances = tf.convert_to_tensor(variances, dtype=self.dtype)

        if means.ndim == 1:
            means = tf.reshape(means, (1, 1, 1, -1))
        if variances.ndim == 1:
            variances = tf.reshape(variances, (1, 1, 1, -1))

        # Build layer if not already built
        if not self.built:
            self.build((None, None, None, means.shape[-1]))

        if means.shape[-1] != self.nb_channels:
            raise ValueError(
                f"New stats have {means.shape[-1]} channels, expected {self.nb_channels}"
            )

        self.mean.assign(means)
        self.variance.assign(variances)

        # Update numpy copies for serialization
        self.means_np = means.numpy()
        self.variances_np = variances.numpy()

    def compute_stats(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        means = tf.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)
        variances = tf.reduce_mean(
            tf.square(inputs - means), axis=[0, 1, 2], keepdims=True
        )

        return means, variances


class StandardizationLayer(tf.keras.layers.Layer):
    """
    Options for normalization strategies in PDE problems.
    """

    def __init__(self, mode="channel", epsilon=1e-6, **kwargs):
        """
        Args:
            mode: 'spatial' - normalize each sample/channel over spatial dims
                  'channel' - normalize each channel over batch and spatial dims
                  'global' - normalize over everything (batch, spatial, channels)
        """
        super(StandardizationLayer, self).__init__(**kwargs)
        self.mode = mode
        self.epsilon = epsilon

    def call(self, inputs: tf.Tensor):

        if self.mode == "spatial":
            # Normalize each sample/channel over spatial dimensions [H, W]
            mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            variance = tf.reduce_mean(
                tf.square(inputs - mean), axis=[1, 2], keepdims=True
            )
        elif self.mode == "channel":
            # Normalize each channel over batch and spatial [N, H, W]
            mean = tf.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)
            variance = tf.reduce_mean(
                tf.square(inputs - mean), axis=[0, 1, 2], keepdims=True
            )

        elif self.mode == "global":
            # Normalize over everything
            mean = tf.reduce_mean(inputs)
            variance = tf.reduce_mean(tf.square(inputs - mean))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        std = tf.sqrt(variance + self.epsilon)
        normalized = (inputs - mean) / std

        return normalized

    def set_stats(self, means, variances):
        pass

    def compute_stats(self, inputs):
        return None, None


class IdentityTransformation(tf.keras.layers.Layer):
    """
    Identity layer that does not change the input at all.
    """

    def __init__(self, **kwargs):

        super(IdentityTransformation, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs
        # return tf.keras.layers.Identity(inputs) # look into this for comparision (as in trainable gradients etc.)

    def set_stats(self, means, variances):
        pass  # fixed does not do anything

    def compute_stats(self, inputs):  # fix liskov violation here
        return None, None
