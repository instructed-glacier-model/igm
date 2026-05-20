import tensorflow as tf
import numpy as np


def parse_cfg_input_names_Nz(cfg, nb_inputs=None, nb_outputs=None):
    """Extract input_names and Nz from cfg with optional sanity checks.

    Helper for the architecture constructor's cfg-positional calling path
    (``Architecture(cfg, nb_inputs, nb_outputs)``).
    """
    input_names = list(cfg.processes.iceflow.unified.inputs)
    Nz = int(cfg.processes.iceflow.numerics.Nz)
    if nb_inputs is not None and len(input_names) != int(nb_inputs):
        raise ValueError(
            f"nb_inputs={nb_inputs} does not match "
            f"len(cfg.processes.iceflow.unified.inputs)={len(input_names)}"
        )
    if nb_outputs is not None and 2 * Nz != int(nb_outputs):
        raise ValueError(
            f"nb_outputs={nb_outputs} does not match "
            f"2 * cfg.processes.iceflow.numerics.Nz = {2 * Nz}"
        )
    return input_names, Nz


def parse_cfg_network_params_strict(cfg, arch_name):
    """Return ``cfg.processes.iceflow.emulator.network.params`` as a dict.

    Raises ``ValueError`` if the node is absent, ``None``, or an empty mapping.
    Used by architectures that have no legacy individual-field fallback.
    """
    network_node = cfg.processes.iceflow.emulator.network
    params_node = getattr(network_node, "params", None)
    params = dict(params_node) if params_node is not None else {}
    if not params:
        raise ValueError(
            f"{arch_name} requires cfg.processes.iceflow.emulator.network.params "
            "to be a non-empty mapping. This architecture has no legacy field fallback."
        )
    return params


class DTypeActivation(tf.keras.layers.Layer):
    """Activation layer that preserves dtype using tf.nn functions (keras was overriding float64 and casting it to float32...)."""

    def __init__(self, activation_name, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.activation_name = activation_name.lower()

        # Map common activation names to tf.nn functions
        self.activation_map = {
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "sigmoid": tf.nn.sigmoid,
            "softmax": tf.nn.softmax,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leakyrelu": tf.nn.leaky_relu,
        }

        if self.activation_name not in self.activation_map:
            raise ValueError(
                f"Activation '{activation_name}' not supported. "
                f"Supported activations: {list(self.activation_map.keys())}"
            )

        self.activation_fn = self.activation_map[self.activation_name]

    def call(self, inputs):
        # Apply activation - tf.nn functions preserve dtype
        return self.activation_fn(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation_name": self.activation_name,
            }
        )
        return config


class PeriodicBCLayer(tf.keras.layers.Layer):
    """
    Fourier feature embedding layer for ENCODING periodic boundary conditions. This does NOT enforce constraints - it just encodes them to the input.
    Based on equation 18 from "Residual-based attention and connection to information bottleneck theory in PINNs"

    Uses 2D Fourier features: [cos(w_x*x)cos(w_y*y), cos(w_x*x)sin(w_y*y),
                                 sin(w_x*x)cos(w_y*y), sin(w_x*x)sin(w_y*y)]
    for multiple frequencies.

    The period is automatically set to 1.0 in normalized coordinates [0, 1] × [0, 1].
    This works because we only care about the periodicity constraint, not the physical scale.
    """

    def __init__(self, num_frequencies=3, name="periodic_bc", **kwargs):
        """
        Args:
            num_frequencies: Number of frequency modes to include (m, n in paper)
            name: Name for the layer (default: 'periodic_bc')
        """
        super(PeriodicBCLayer, self).__init__(name=name, **kwargs)
        self.num_frequencies = num_frequencies

        # Use normalized period [0, 1] for both dimensions
        # This is sufficient because periodicity is scale-invariant
        self.period_x = 1.0
        self.period_y = 1.0

        # Compute angular frequencies: w_x = 2π/P_x = 2π, w_y = 2π/P_y = 2π
        self.w_x = tf.constant(2.0 * np.pi / self.period_x, dtype=self.dtype)
        self.w_y = tf.constant(2.0 * np.pi / self.period_y, dtype=self.dtype)

    def call(self, inputs):
        # inputs: (batch, height, width, channels)
        # All dimensions can be None/dynamic at graph construction time
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Create coordinate grids in normalized [0, 1] space
        # This is computed dynamically based on actual input shape
        x_coords = tf.linspace(0.0, self.period_x, width)
        y_coords = tf.linspace(0.0, self.period_y, height)

        # Create meshgrid: (height, width)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing="ij")

        # Expand to batch dimension: (1, height, width)
        xx = tf.expand_dims(xx, 0)
        yy = tf.expand_dims(yy, 0)

        # Tile to match batch size: (batch, height, width)
        xx = tf.tile(xx, [batch_size, 1, 1])
        yy = tf.tile(yy, [batch_size, 1, 1])

        xx = tf.cast(xx, dtype=self.dtype)
        yy = tf.cast(yy, dtype=self.dtype)

        # Build Fourier features
        fourier_features = []

        for m in range(self.num_frequencies):
            for n in range(self.num_frequencies):
                # Skip m=0, n=0 (constant term already in input)
                if m == 0 and n == 0:
                    continue

                # Compute frequencies
                w_mx = m * self.w_x
                w_ny = n * self.w_y

                # Four combinations from equation 18:
                # cos(w_x*x)cos(w_y*y)
                fourier_features.append(
                    tf.expand_dims(tf.cos(w_mx * xx) * tf.cos(w_ny * yy), -1)
                )
                # cos(w_x*x)sin(w_y*y)
                fourier_features.append(
                    tf.expand_dims(tf.cos(w_mx * xx) * tf.sin(w_ny * yy), -1)
                )
                # sin(w_x*x)cos(w_y*y)
                fourier_features.append(
                    tf.expand_dims(tf.sin(w_mx * xx) * tf.cos(w_ny * yy), -1)
                )
                # sin(w_x*x)sin(w_y*y)
                fourier_features.append(
                    tf.expand_dims(tf.sin(w_mx * xx) * tf.sin(w_ny * yy), -1)
                )

        # Concatenate all Fourier features: (batch, height, width, num_fourier_features)
        fourier_features = tf.concat(fourier_features, axis=-1)

        # Concatenate with original input
        # Result: (batch, height, width, channels + num_fourier_features)
        augmented_input = tf.concat([inputs, fourier_features], axis=-1)

        return augmented_input

    def get_config(self):
        config = super(PeriodicBCLayer, self).get_config()
        config.update({"num_frequencies": self.num_frequencies})
        return config

    def compute_output_shape(self, input_shape):
        # Calculate number of Fourier features
        # 4 features per (m,n) pair, excluding (0,0)
        num_features = 4 * (self.num_frequencies**2 - 1)
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3] + num_features,
        )


class PeriodicBCFourier(tf.keras.layers.Layer):
    """
    Enforces periodic boundary conditions by projecting output into Fourier space.
    This ensures perfect periodicity by construction.
    """

    def __init__(self, name="periodic_bc_fourier", **kwargs):
        super(PeriodicBCFourier, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # inputs: (batch, height, width, channels)
        # Note: This assumes height and width are powers of 2 for efficiency

        # Move channels to front for FFT
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = tf.transpose(inputs, [0, 3, 1, 2])

        # Cast to complex
        x_complex = tf.cast(
            x, tf.complex64
        )  # ! make this 2x what the real precision is (i.e if inputs are float64, this should be complex128 etc.)

        # 2D FFT
        x_fft = tf.signal.fft2d(x_complex)

        # Inverse FFT (this inherently enforces periodicity)
        x_ifft = tf.signal.ifft2d(x_fft)

        # Take real part
        x_real = tf.math.real(x_ifft)

        # Transpose back
        output = tf.transpose(x_real, [0, 2, 3, 1])

        return output


class PeriodicBCAnsatz(tf.keras.layers.Layer):
    """
    Enforces periodic boundary conditions using a smooth ansatz function.

    The output is multiplied by a weighting function that goes to zero at boundaries,
    plus a correction term that ensures periodicity.

    u_periodic(x,y) = u_nn(x,y) * w(x) * w(y) + boundary_correction(x,y)

    where w(x) = sin^2(πx) smoothly goes to 0 at x=0 and x=1
    """

    def __init__(self, name="periodic_bc_ansatz", **kwargs):
        super(PeriodicBCAnsatz, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # inputs: (batch, height, width, channels)
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        # Create normalized coordinates [0, 1]
        x_coords = tf.linspace(0.0, 1.0, width)
        y_coords = tf.linspace(0.0, 1.0, height)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing="ij")

        # Expand dimensions: (1, height, width, 1)
        xx = tf.reshape(xx, [1, height, width, 1])
        yy = tf.reshape(yy, [1, height, width, 1])

        # Weight function: sin^2(πx) * sin^2(πy)
        # This is 0 at boundaries and 1 at center
        w_x = tf.sin(np.pi * xx) ** 2
        w_y = tf.sin(np.pi * yy) ** 2
        weight = w_x * w_y

        # Apply weighting to network output
        # This forces the output toward zero at boundaries
        weighted_output = inputs * weight

        # Extract boundary values to create periodic correction
        # Average boundaries to ensure they match
        left = inputs[:, :, 0:1, :]
        right = inputs[:, :, -1:, :]
        top = inputs[:, 0:1, :, :]
        bottom = inputs[:, -1:, :, :]

        # Boundary averages
        left_right_avg = (left + right) / 2.0
        top_bottom_avg = (top + bottom) / 2.0

        # Create smooth interpolation of boundary values
        # Linear interpolation from left to right
        boundary_lr = left_right_avg * (1 - xx) + left_right_avg * xx
        # Linear interpolation from top to bottom
        boundary_tb = top_bottom_avg * (1 - yy) + top_bottom_avg * yy

        # Combine: use inverse weight to add boundary correction
        boundary_correction = boundary_lr * (1 - w_x) + boundary_tb * (1 - w_y)

        # Final output with periodic BCs enforced
        output = weighted_output + boundary_correction

        return output


class PeriodicBCEnforcement(tf.keras.layers.Layer):
    """
    Enforces periodic boundary conditions on output by averaging boundary values.
    This ensures that output[0, :] == output[-1, :] and output[:, 0] == output[:, -1]
    """

    def __init__(self, name="periodic_bc_enforcement", **kwargs):
        super(PeriodicBCEnforcement, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # inputs: (batch, height, width, channels)

        # Average left and right boundaries
        left_right_avg = (inputs[:, :, 0:1, :] + inputs[:, :, -1:, :]) / 2.0

        # Average top and bottom boundaries
        top_bottom_avg = (inputs[:, 0:1, :, :] + inputs[:, -1:, :, :]) / 2.0

        # Average corners (all four corners should be equal)
        corner_avg = (
            inputs[:, 0:1, 0:1, :]
            + inputs[:, 0:1, -1:, :]
            + inputs[:, -1:, 0:1, :]
            + inputs[:, -1:, -1:, :]
        ) / 4.0

        # Reconstruct the output with enforced periodicity
        # Interior stays the same
        output = inputs

        # Set left and right boundaries
        output = tf.concat(
            [left_right_avg, output[:, :, 1:-1, :], left_right_avg], axis=2
        )

        # Set top and bottom boundaries
        output = tf.concat(
            [top_bottom_avg, output[:, 1:-1, :, :], top_bottom_avg], axis=1
        )

        # Set corners to average of all corners
        # Top-left
        output = tf.concat(
            [
                tf.concat([corner_avg, output[:, 0:1, 1:, :]], axis=2),
                output[:, 1:, :, :],
            ],
            axis=1,
        )

        # Top-right (already handled by left-right averaging)
        # Bottom corners (already handled by top-bottom averaging)

        return output
