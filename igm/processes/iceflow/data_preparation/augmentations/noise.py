import tensorflow as tf
from .base import Augmentation


class NoiseParams(tf.experimental.ExtensionType):
    noise_type: str
    noise_scale: float
    channel_mask: tf.Tensor  # Boolean mask indicating which channels to apply noise to


class NoiseAugmentation(Augmentation):
    def __init__(self, params: NoiseParams):
        self.params = params

    def apply(self, x):
        return add_noise_selective(
            x, self.params.noise_type, self.params.noise_scale, self.params.channel_mask
        )


# Create a registry of noise functions for cleaner dispatch
NOISE_FUNCTIONS = {
    "gaussian": lambda x, scale: add_gaussian_noise(x, scale),
    "perlin": lambda x, scale: add_perlin_noise(x, scale),
    "intensity": lambda x, scale: add_intensity_noise(x, scale),
    "none": lambda x, scale: x,
}


@tf.function
def add_noise_selective(x, noise_type, noise_scale, channel_mask):
    """Apply noise selectively to specific channels based on mask."""
    # First apply noise to the entire tensor
    noisy_x = add_noise(x, noise_type, noise_scale)

    # Create a mask that matches the tensor shape
    # channel_mask should be shape [num_channels] with boolean values
    mask_expanded = tf.reshape(channel_mask, [1, 1, -1])  # Shape: [1, 1, channels]
    mask_expanded = tf.cast(mask_expanded, x.dtype)

    # Apply noise only to selected channels, keep original values for others
    return x * (1 - mask_expanded) + noisy_x * mask_expanded


@tf.function
def add_noise(x, noise_type, noise_scale):
    """Apply noise based on type using a more elegant dispatch pattern."""
    # Use tf.case for cleaner conditional logic
    return tf.case(
        [
            (
                tf.equal(noise_type, "gaussian"),
                lambda: add_gaussian_noise(x, noise_scale),
            ),
            (tf.equal(noise_type, "perlin"), lambda: add_perlin_noise(x, noise_scale)),
            (
                tf.equal(noise_type, "intensity"),
                lambda: add_intensity_noise(x, noise_scale),
            ),
        ],
        default=lambda: x,  # Default case for "none" or unknown types
        exclusive=True,
    )


@tf.function
def add_intensity_noise(x, noise_scale):
    noise_value = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=x.dtype)
    noise_tensor = tf.fill(tf.shape(x), noise_value)
    return x * (1.0 + noise_scale * noise_tensor)


@tf.function
def add_gaussian_noise(x, noise_scale):
    noise = tf.random.normal(tf.shape(x), stddev=1.0)
    return x + noise_scale * noise


@tf.function
def add_perlin_noise(x, noise_scale, base_resolution=4, octaves=2, persistence=0.5):
    height = tf.shape(x)[0]
    width = tf.shape(x)[1]
    n_channels = tf.shape(x)[-1]
    dtype = x.dtype

    def octave_body(i, amplitude, max_amplitude, total_noise):
        freq = tf.cast(base_resolution, tf.int32) * (2**i)
        resolution = (freq, freq)
        angles = tf.random.uniform(
            shape=(resolution[0] + 1, resolution[1] + 1),
            minval=0,
            maxval=2 * tf.constant(3.141592653589793, dtype=dtype),
            dtype=dtype,
        )
        gradients_x = tf.cos(angles)
        gradients_y = tf.sin(angles)
        gradients_tf = tf.stack([gradients_x, gradients_y], axis=-1)

        def perlin_noise_2d(_):
            d0 = height // resolution[0]
            d1 = width // resolution[1]
            delta0 = tf.cast(resolution[0], dtype) / tf.cast(height, dtype)
            delta1 = tf.cast(resolution[1], dtype) / tf.cast(width, dtype)
            x_indices = tf.range(height, dtype=dtype)
            y_indices = tf.range(width, dtype=dtype)
            xx = x_indices * delta0
            yy = y_indices * delta1
            xx, yy = tf.meshgrid(xx, yy, indexing="ij")
            grid = tf.stack([xx % 1.0, yy % 1.0], axis=-1)
            gradients_rep = tf.repeat(tf.repeat(gradients_tf, d0, axis=0), d1, axis=1)
            g00 = gradients_rep[:-d0, :-d1]
            g10 = gradients_rep[d0:, :-d1]
            g01 = gradients_rep[:-d0, d1:]
            g11 = gradients_rep[d0:, d1:]
            offset_00 = tf.stack([grid[:, :, 0], grid[:, :, 1]], axis=-1)
            offset_10 = tf.stack([grid[:, :, 0] - 1, grid[:, :, 1]], axis=-1)
            offset_01 = tf.stack([grid[:, :, 0], grid[:, :, 1] - 1], axis=-1)
            offset_11 = tf.stack([grid[:, :, 0] - 1, grid[:, :, 1] - 1], axis=-1)
            n00 = tf.reduce_sum(offset_00 * g00, axis=-1)
            n10 = tf.reduce_sum(offset_10 * g10, axis=-1)
            n01 = tf.reduce_sum(offset_01 * g01, axis=-1)
            n11 = tf.reduce_sum(offset_11 * g11, axis=-1)
            t_vals = grid
            t = t_vals * t_vals * t_vals * (t_vals * (t_vals * 6 - 15) + 10)
            n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
            n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
            result = tf.cast(tf.sqrt(2.0), dtype) * (
                (1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1
            )
            return result

        perlin_noise = tf.map_fn(
            perlin_noise_2d,
            tf.range(n_channels),
            fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=dtype),
        )
        perlin_noise = tf.transpose(perlin_noise, [1, 2, 0])
        total_noise = total_noise + amplitude * perlin_noise
        max_amplitude = max_amplitude + amplitude
        amplitude = amplitude * persistence
        return i + 1, amplitude, max_amplitude, total_noise

    i0 = tf.constant(0)
    amplitude0 = tf.constant(1.0, dtype=dtype)
    max_amplitude0 = tf.constant(0.0, dtype=dtype)
    total_noise0 = tf.zeros_like(x, dtype=dtype)

    def cond(i, amplitude, max_amplitude, total_noise):
        return i < octaves

    _, _, max_amplitude, total_noise = tf.while_loop(
        cond, 
        octave_body, 
        loop_vars=[i0, amplitude0, max_amplitude0, total_noise0],
        shape_invariants=[
            tf.TensorShape([]),  # i
            tf.TensorShape([]),  # amplitude  
            tf.TensorShape([]),  # max_amplitude
            tf.TensorShape([None, None, None])  # total_noise - allow variable shape
        ]
    )

    # Normalize by max_amplitude to account for octave scaling
    total_noise = total_noise / max_amplitude

    # Additional normalization to ensure range is always [-1, 1]
    # The Perlin noise function has an inherent range factor of sqrt(2)
    total_noise = total_noise / tf.cast(tf.sqrt(2.0), dtype)

    return x + noise_scale * total_noise
