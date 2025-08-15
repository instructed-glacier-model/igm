import tensorflow as tf

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .patching import OverlapPatching


# Define PreparationParams as a tf.experimental.ExtensionType
class PreparationParams(tf.experimental.ExtensionType):
    overlap: float  # overlap is the fraction of the patch size that overlaps between adjacent patches (0.0 to 1.0)
    batch_size: int  # batch size for training
    patch_size: int  # size of the patches to extract from the images (e.g. 64 results in 64x64 patches)
    rotation_probability: float  # probability of rotating each training image
    flip_probability: float  # probability of flipping each training image
    noise_type: str  # type of noise to add (e.g. "gaussian", "perlin" or "none")
    noise_scale: float  # maximum fractional noise scale e.g. noise_scale = 0.2 means max +/- 20% noise added
    target_samples: int  # target total number of training images to generate


def create_training_set(
    input: tf.Tensor, preparation_params: PreparationParams
) -> tf.data.Dataset:
    # Detect the dtype from the input tensor
    dtype = input.dtype

    Patching = OverlapPatching(
        patch_size=preparation_params.patch_size, overlap=preparation_params.overlap
    )
    patches = Patching.patch_tensor(input)
    patches = tf.cast(patches, dtype)
    patch_shape = tf.shape(patches)
    Ny = patch_shape[1]
    Nx = patch_shape[2]

    dataset = tf.data.Dataset.from_tensor_slices(patches)

    # Apply all augmentations in a single map operation for better performance
    dataset = dataset.map(
        lambda x: _apply_all_augmentations(
            x,
            preparation_params.rotation_probability,
            preparation_params.flip_probability,
            preparation_params.noise_type,
            preparation_params.noise_scale,
            dtype,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle before upsampling
    dataset = dataset.shuffle(buffer_size=1000)

    # Upsample with repeat and take to reach exactly target_samples
    dataset = _upsample_dataset(
        dataset,
        patch_shape,
        preparation_params.target_samples,
        dtype,
    )

    # Batch and prefetch
    dataset = dataset.batch(preparation_params.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, Ny, Nx


def get_available_gpu_memory(default_gb=1):
    """Return available GPU memory (bytes), default to default_gb if no GPU or error."""
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        if not gpu_devices:
            return int(default_gb * 1024**3)

        # Try to get memory info first
        gpu_memory_info = tf.config.experimental.get_memory_info("GPU:0")
        memory_limit = gpu_memory_info.get("limit", None)
        current_usage = gpu_memory_info.get("current", 0)

        if memory_limit is not None:
            # Memory limit is set, use it
            available_memory = memory_limit - current_usage
            safe_memory = int(available_memory * 0.8)
            return safe_memory
        else:
            # No memory limit set, estimate from GPU specs
            # Use a conservative estimate based on detected GPU (5563 MB from logs)
            estimated_total_gb = 5.5  # Based on the 5563 MB we saw in logs
            estimated_total_bytes = int(estimated_total_gb * 1024**3)
            safe_memory = int(estimated_total_bytes * 0.6)  # Conservative 60%
            return safe_memory

    except Exception:
        return int(default_gb * 1024**3)


def calculate_bytes_per_patch(patch_shape, dtype):
    """Calculate memory usage per patch in bytes."""

    if dtype == tf.float64:
        bytes_per_element = 8
    elif dtype == tf.float32:
        bytes_per_element = 4
    elif dtype == tf.float16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4  # Default to float32
    elements_per_patch = (
        patch_shape[0] * patch_shape[1] * patch_shape[2]
        if len(patch_shape) >= 3
        else tf.reduce_prod(patch_shape)
    )
    return elements_per_patch * bytes_per_element


def _upsample_dataset(
    dataset: tf.data.Dataset,
    patch_shape: tuple,
    target_samples: int,
    dtype: tf.DType = None,
) -> tf.data.Dataset:
    """Upsample dataset to reach target number of samples."""

    adjusted_target_samples = target_samples

    # ================= SKIP THIS CHECK FOR NOW... FIX LATER ========================================

    # bytes_per_patch = calculate_bytes_per_patch(patch_shape[1:], dtype)

    # safe_memory = get_available_gpu_memory()

    # # Calculate maximum samples that fit in memory
    # max_samples_in_memory = safe_memory // bytes_per_patch
    # # Adjust target_samples if it exceeds memory limit
    # adjusted_target_samples = min(target_samples, max_samples_in_memory)

    # if adjusted_target_samples < target_samples:
    #     print(
    #         f"Warning: Reducing target_samples from {target_samples} to {adjusted_target_samples} to fit in GPU memory"
    #     )
    #     print(
    #         f"Memory per patch: {bytes_per_patch / 1024**2:.2f} MB, Safe GPU memory: {safe_memory / 1024**3:.2f} GB"
    #     )

    num_patches = patch_shape[0]

    if num_patches < adjusted_target_samples:
        repeats = tf.cast(
            tf.math.ceil(
                tf.cast(adjusted_target_samples, tf.float32)
                / tf.cast(num_patches, tf.float32)
            ),
            tf.int64,
        )
        dataset = dataset.repeat(repeats)
    return dataset.take(tf.cast(adjusted_target_samples, tf.int64))


@tf.function
def _apply_all_augmentations(
    x, rotation_probability, flip_probability, noise_type, noise_scale, dtype
):
    rotation_params = RotationParams(probability=rotation_probability)
    flip_params = FlipParams(probability=flip_probability)
    noise_params = NoiseParams(noise_type=noise_type, noise_scale=noise_scale)
    augmentations = [
        RotationAugmentation(rotation_params),
        FlipAugmentation(flip_params),
        NoiseAugmentation(noise_params),
    ]
    for aug in augmentations:
        x = aug.apply(x)
    return tf.cast(x, dtype)
