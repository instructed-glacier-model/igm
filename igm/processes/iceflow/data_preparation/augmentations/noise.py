import tensorflow as tf
from .base import Augmentation


class NoiseParams(tf.experimental.ExtensionType):
    noise_type: str
    noise_scale: float
    channel_mask: tf.Tensor  # Boolean mask for which channels get noise


class NoiseAugmentation(Augmentation):
    def __init__(self, params: NoiseParams):
        self.params = params

    def apply(self, x):
        return add_noise_selective(
            x, self.params.noise_type, self.params.noise_scale, self.params.channel_mask
        )


# Noise function registry
NOISE_FUNCTIONS = {
    "gaussian": lambda x, scale: add_gaussian_noise(x, scale),
    "perlin": lambda x, scale: add_perlin_noise(x, scale),
    "intensity": lambda x, scale: add_intensity_noise(x, scale),
    "none": lambda x, scale: x,
}


@tf.function
def add_noise_selective(x, noise_type, noise_scale, channel_mask):
    """Apply noise to only the channels specified by the mask."""
    noisy_x = add_noise(x, noise_type, noise_scale)

    # Expand mask to match tensor dimensions
    mask_expanded = tf.reshape(channel_mask, [1, 1, -1])
    mask_expanded = tf.cast(mask_expanded, x.dtype)

    # Mix original and noisy versions based on mask
    return x * (1 - mask_expanded) + noisy_x * mask_expanded


@tf.function
def add_noise(x, noise_type, noise_scale):
    """Dispatch noise application based on type."""
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
        default=lambda: x,
        exclusive=True,
    )


@tf.function
def generate_intensity_noise(shape, dtype):
    """
    Generate uniform noise with same value across all pixels.

    Args:
        shape: Output tensor shape
        dtype: Data type

    Returns:
        Tensor filled with single random value in [-1, 1]
    """
    noise_value = tf.random.uniform([], minval=-1.0, maxval=1.0, dtype=dtype)
    return tf.fill(shape, noise_value)


@tf.function
def generate_gaussian_noise(shape, dtype, num_uniforms: int = 6):
    """
    Generate bell-shaped noise by summing uniform random variables.

    Uses central limit theorem to approximate Gaussian distribution while
    guaranteeing values stay in [-1, 1] range.

    Args:
        shape: Output tensor shape
        dtype: Data type
        num_uniforms: Number of uniform variables to sum (higher = more bell-shaped)

    Returns:
        Bell-shaped noise tensor with values in [-1, 1]
    """
    u = tf.random.uniform(
        tf.concat([shape, [num_uniforms]], axis=0),
        minval=-1.0 / tf.cast(num_uniforms, dtype),
        maxval=1.0 / tf.cast(num_uniforms, dtype),
        dtype=dtype,
    )
    return tf.reduce_sum(u, axis=-1)


@tf.function
def apply_multiplicative_noise(x, noise, noise_scale):
    """
    Apply noise multiplicatively: result = x * (1 + noise_scale * noise).

    Args:
        x: Input tensor
        noise: Noise tensor with same shape as x
        noise_scale: Scaling factor for noise intensity

    Returns:
        Input tensor with multiplicative noise applied
    """
    return x * (1.0 + noise_scale * noise)


@tf.function
def add_intensity_noise(x, noise_scale):
    """Add uniform global scaling noise to input."""
    noise = generate_intensity_noise(tf.shape(x), x.dtype)
    return apply_multiplicative_noise(x, noise, noise_scale)


@tf.function
def add_gaussian_noise(x, noise_scale):
    """Add bell-shaped random noise to input."""
    noise = generate_gaussian_noise(tf.shape(x), x.dtype)
    return apply_multiplicative_noise(x, noise, noise_scale)


@tf.function
def generate_perlin_noise(shape, dtype, base_resolution=4, octaves=3, persistence=0.6):
    """
    Generate spatially coherent Perlin noise.

    Creates smooth, natural-looking noise patterns by combining multiple
    octaves of procedural noise at different frequencies.

    Args:
        shape: Output tensor shape [height, width, channels]
        dtype: Data type
        base_resolution: Grid resolution for first octave
        octaves: Number of noise octaves to combine
        persistence: Amplitude scaling between octaves

    Returns:
        Spatially coherent noise tensor with values in [-1, 1]
    """
    height = shape[0]
    width = shape[1]
    n_channels = shape[-1]

    octaves = tf.convert_to_tensor(octaves, dtype=tf.int32)
    persistence = tf.cast(persistence, dtype)

    def octave_body(i, amplitude, total_noise):
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

            # Calculate offset vectors from grid points to gradients
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
            # Bilinear interpolation between corner values
            result = (1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1
            return result

        perlin_noise = tf.map_fn(
            perlin_noise_2d,
            tf.range(n_channels),
            fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=dtype),
        )
        perlin_noise = tf.transpose(perlin_noise, [1, 2, 0])
        total_noise = total_noise + amplitude * perlin_noise
        amplitude = amplitude * persistence
        return i + 1, amplitude, total_noise

    i0 = tf.constant(0)
    amplitude0 = tf.cast(0.8, dtype=dtype)
    total_noise0 = tf.zeros(shape, dtype=dtype)

    def cond(i, amplitude, total_noise):
        return i < octaves

    _, _, total_noise = tf.while_loop(
        cond,
        octave_body,
        loop_vars=[i0, amplitude0, total_noise0],
        shape_invariants=[
            tf.TensorShape([]),  # i
            tf.TensorShape([]),  # amplitude
            tf.TensorShape([None, None, None]),  # total_noise
        ],
    )

    # Scale and bound output to maximize use of [-1, 1] range
    total_noise = tf.tanh(total_noise * 1.2)

    return total_noise


@tf.function
def add_perlin_noise(x, noise_scale, base_resolution=4, octaves=2, persistence=0.5):
    """Add spatially coherent Perlin noise to input."""
    noise = generate_perlin_noise(
        tf.shape(x), x.dtype, base_resolution, octaves, persistence
    )
    return apply_multiplicative_noise(x, noise, noise_scale)
