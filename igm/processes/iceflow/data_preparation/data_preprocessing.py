import tensorflow as tf
from typing import Any, Dict, Tuple

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .patching import OverlapPatching
from igm.processes.iceflow.utils import fieldin_to_X_2d, fieldin_to_X_3d


class PreparationParams(tf.experimental.ExtensionType):
    overlap: float  # overlap is the fraction of the patch size that overlaps between adjacent patches (0.0 to 1.0)
    batch_size: int  # batch size for training
    patch_size: int  # size of the patches to extract from the images (e.g. 64 results in 64x64 patches)
    rotation_probability: float  # probability of rotating each training image
    flip_probability: float  # probability of flipping each training image
    noise_type: str  # type of noise to add (e.g. "gaussian", "perlin" or "none")
    noise_scale: float  # maximum fractional noise scale e.g. noise_scale = 0.2 means max +/- 20% noise added
    target_samples: int  # target total number of training images to generate
    fieldin_names: Tuple[
        str, ...
    ]  # names of input fields for selective noise application
    noise_channels: Tuple[str, ...]  # names of input fields to apply noise


def _determine_noise_channels(cfg) -> Tuple[str, ...]:
    """
    Determine which channels should have noise applied based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        Tuple[str, ...]: Channel names to apply noise to
    """
    if hasattr(cfg.processes, "data_assimilation"):
        # Use control_list fields that are suitable for noise
        noise_channels = []
        for f in cfg.processes.data_assimilation.control_list:
            if f in ["thk", "usurf"]:  # Only apply noise to these fields
                noise_channels.append(f)
        # Convert to tuple for tf.experimental.ExtensionType compatibility
        return tuple(noise_channels) if noise_channels else ("thk", "usurf")
    else:
        # Default noise channels
        return ("thk", "usurf")


def get_input_params_args(cfg) -> Dict[str, Any]:

    # return {
    #     "overlap": cfg_emulator.overlap,
    #     "batch_size": cfg_emulator.batch_size,
    #     "patch_size": cfg_emulator.patch_size,
    #     "rotation_probability": cfg_emulator.rotation_probability,
    #     "flip_probability": cfg_emulator.flip_probability,
    #     "noise_type": cfg_emulator.noise_type,
    #     "noise_scale": cfg_emulator.noise_scale,
    #     "target_samples": cfg_emulator.target_samples,
    # }

    return {
        "overlap": 0.25,
        "batch_size": 16,
        "patch_size": 64,
        "rotation_probability": 0.0,
        "flip_probability": 0.0,
        "noise_type": "perlin",
        "noise_scale": 0.1,
        "target_samples": 32,
        "fieldin_names": cfg.processes.iceflow.emulator.fieldin,
        "noise_channels": _determine_noise_channels(cfg),
    }


def _handle_small_input_case(
    input_height: int,
    input_width: int,
    patch_size: int,
    batch_size: int,
    target_samples: int,
) -> tuple:
    """
    Handle the case where input is smaller than patch size.

    Args:
        input_height: Height of input tensor
        input_width: Width of input tensor
        patch_size: Size of patches
        batch_size: Batch size
        target_samples: Target number of samples

    Returns:
        tuple: (input_height, input_width, effective_batch_size)
    """
    return (
        input_height,
        input_width,
        min(batch_size, max(target_samples, 1)),
    )


def scale_inputs(cfg, X: tf.Tensor) -> tf.Tensor:
    """
    Apply input scaling to a tensor based on configuration.

    Args:
        cfg: Configuration object
        X: Input tensor to scale

    Returns:
        tf.Tensor: Scaled tensor
    """
    try:
        # Try to get scaling factors from configuration
        inputs_scales_list = cfg.processes.iceflow.unified.inputs_scales
    except AttributeError:
        # If not configured, use scaling factor of 1 for all channels
        num_channels = tf.shape(X)[-1]
        inputs_scales_list = [1.0] * num_channels

    # Convert to tensor for TensorFlow operations
    scales = tf.constant(inputs_scales_list, dtype=X.dtype)

    # Apply scaling by dividing each channel by its corresponding scale
    scaled_X = X / scales

    return scaled_X


def _calculate_patch_counts(
    input_height: int, input_width: int, patch_size: int, overlap: float
) -> tuple:
    """
    Calculate number of patches in each direction.

    Args:
        input_height: Height of input tensor
        input_width: Width of input tensor
        patch_size: Size of patches
        overlap: Overlap fraction

    Returns:
        tuple: (n_patches_y, n_patches_x, total_patches)
    """
    import math

    height_f = float(input_height)
    width_f = float(input_width)
    patch_size_f = float(patch_size)

    # Calculate minimum stride and number of patches
    min_stride = int(patch_size_f * (1.0 - overlap))

    # Calculate number of patches in each direction
    n_patches_y = max(
        1, int(math.ceil((height_f - patch_size_f) / float(min_stride))) + 1
    )
    n_patches_x = max(
        1, int(math.ceil((width_f - patch_size_f) / float(min_stride))) + 1
    )

    total_patches = n_patches_y * n_patches_x
    return n_patches_y, n_patches_x, total_patches


def _calculate_final_dimensions(
    patch_size: int, num_patches: int, batch_size: int, target_samples: int
) -> tuple:
    """
    Calculate final dimensions and batch size.

    Args:
        patch_size: Size of patches
        num_patches: Total number of patches
        batch_size: Requested batch size
        target_samples: Target number of samples

    Returns:
        tuple: (Ny, Nx, effective_batch_size)
    """
    # Patch dimensions are always patch_size x patch_size
    Ny = patch_size
    Nx = patch_size

    # Calculate adjusted target samples (must be at least num_patches)
    adjusted_target_samples = max(target_samples, num_patches)

    # Calculate effective batch size (min of batch_size and adjusted_target_samples)
    effective_batch_size = min(batch_size, adjusted_target_samples)

    return Ny, Nx, effective_batch_size


def calculate_expected_dimensions(
    input_height: int,
    input_width: int,
    preparation_params: PreparationParams,
) -> tuple:
    """
    Calculate the expected Ny, Nx, num_patches, and effective_batch_size for given input dimensions.
    This function precisely matches the behavior of the data preprocessing pipeline.

    Args:
        input_height (int): Height of the input tensor in pixels.
        input_width (int): Width of the input tensor in pixels.
        preparation_params (PreparationParams): Parameters containing patch_size, overlap, batch_size, target_samples.

    Returns:
        tuple: (Ny, Nx, num_patches, effective_batch_size, adjusted_target_samples)
            - Ny: Patch height (same as patch_size)
            - Nx: Patch width (same as patch_size)
            - num_patches: Total number of patches extracted from input
            - effective_batch_size: Actual batch size used (min of batch_size and adjusted_target_samples)
            - adjusted_target_samples: Target samples after adjustment (max of target_samples and num_patches)
    """
    patch_size = preparation_params.patch_size
    overlap = preparation_params.overlap
    batch_size = preparation_params.batch_size
    target_samples = preparation_params.target_samples

    # Handle case where input is smaller than patch size
    if patch_size > input_width and patch_size > input_height:
        return _handle_small_input_case(
            input_height, input_width, patch_size, batch_size, target_samples
        )

    # Calculate patch counts and final dimensions
    n_patches_y, n_patches_x, num_patches = _calculate_patch_counts(
        input_height, input_width, patch_size, overlap
    )

    return _calculate_final_dimensions(
        patch_size, num_patches, batch_size, target_samples
    )


def _convert_fieldin_to_tensor(cfg, fieldin) -> tf.Tensor:
    """
    Convert fieldin to tensor based on dimensionality configuration.

    Args:
        cfg: Configuration object containing physics parameters
        fieldin: Input field data

    Returns:
        tf.Tensor: Converted tensor
    """
    dim_arrhenius = cfg.processes.iceflow.physics.dim_arrhenius

    if dim_arrhenius == 3:
        X = fieldin_to_X_3d(dim_arrhenius, fieldin)
    elif dim_arrhenius == 2:
        X = fieldin_to_X_2d(fieldin)
    else:
        raise ValueError(f"Unsupported dim_arrhenius value: {dim_arrhenius}")

    return tf.squeeze(X, axis=0)  # temporary fix necessary for backwards compatibility


def split_fieldin_to_patches(cfg, fieldin, patching: OverlapPatching) -> tuple:
    """
    Create base patches without augmentation (static preprocessing).
    This only needs to be done once and can be reused across epochs.
    """
    X = _convert_fieldin_to_tensor(cfg, fieldin)

    # Apply input scaling
    # X = scale_inputs(cfg, X)

    patches = patching.patch_tensor(X)
    return patches


def _should_apply_augmentations(preparation_params: PreparationParams) -> bool:
    """
    Determine if any augmentations should be applied based on parameters.

    Args:
        preparation_params: Parameters containing augmentation settings

    Returns:
        bool: True if any augmentation should be applied
    """
    has_rotation = preparation_params.rotation_probability > 0
    has_flip = preparation_params.flip_probability > 0
    has_noise = (
        preparation_params.noise_type != "none" and preparation_params.noise_scale > 0
    )
    return has_rotation or has_flip or has_noise


def _apply_augmentations_to_dataset(
    dataset: tf.data.Dataset, preparation_params: PreparationParams, dtype: tf.DType
) -> tf.data.Dataset:
    """
    Apply augmentations to the dataset if they should be applied.

    Args:
        dataset: Input dataset
        preparation_params: Parameters containing augmentation settings
        dtype: Data type for casting

    Returns:
        tf.data.Dataset: Dataset with augmentations applied
    """
    if _should_apply_augmentations(preparation_params):
        return dataset.map(
            lambda x: _apply_all_augmentations(
                x,
                preparation_params.rotation_probability,
                preparation_params.flip_probability,
                preparation_params.noise_type,
                preparation_params.noise_scale,
                dtype,
                preparation_params.fieldin_names,
                preparation_params.noise_channels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset


def _should_upsample_dataset(
    preparation_params: PreparationParams, num_patches: int, apply_augmentations: bool
) -> bool:
    """
    Determine if upsampling should be performed.

    Args:
        preparation_params: Parameters containing target samples
        num_patches: Current number of patches
        apply_augmentations: Whether augmentations are being applied

    Returns:
        bool: True if upsampling should be performed
    """
    should_upsample = preparation_params.target_samples > num_patches

    if should_upsample and not apply_augmentations:
        import warnings

        warnings.warn(
            f"Warning: target_samples ({preparation_params.target_samples}) is greater than "
            f"number of patches ({num_patches}) but no augmentations are enabled. "
            f"Upsampling without augmentations will create identical copies. "
            f"Skipping upsampling and using original {num_patches} samples.",
            UserWarning,
        )
        return False

    return should_upsample


def _calculate_effective_batch_size(
    preparation_params: PreparationParams, adjusted_target_samples: int
) -> int:
    """
    Calculate the effective batch size based on parameters and available samples.

    Args:
        preparation_params: Parameters containing batch size
        adjusted_target_samples: Number of samples after adjustment

    Returns:
        int: Effective batch size to use
    """
    return min(preparation_params.batch_size, adjusted_target_samples)


def _configure_final_dataset(
    dataset: tf.data.Dataset, batch_size: int
) -> tf.data.Dataset:
    """
    Configure the final dataset with batching and prefetching.

    Args:
        dataset: Input dataset
        batch_size: Batch size to use

    Returns:
        tf.data.Dataset: Final configured dataset
    """
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_training_set_from_patches(
    patches: tf.Tensor, preparation_params: PreparationParams
) -> tf.data.Dataset:
    """
    Create a training dataset from pre-computed patches with fresh augmentations.
    This can be called multiple times to get different augmentations.
    """
    patch_shape = tf.shape(patches)
    dtype = patches.dtype
    num_patches = int(patch_shape[0])

    # Create base dataset
    dataset = tf.data.Dataset.from_tensor_slices(patches)

    # Apply augmentations if needed
    apply_augmentations = _should_apply_augmentations(preparation_params)
    dataset = _apply_augmentations_to_dataset(dataset, preparation_params, dtype)

    # Shuffle before upsampling
    dataset = dataset.shuffle(buffer_size=100)

    # Handle upsampling
    should_upsample = _should_upsample_dataset(
        preparation_params, num_patches, apply_augmentations
    )

    if should_upsample:
        dataset, adjusted_target_samples = _upsample_dataset(
            dataset,
            patch_shape,
            preparation_params.target_samples,
            dtype,
        )
    else:
        adjusted_target_samples = num_patches

    # Calculate batch size and configure final dataset
    batch_size = _calculate_effective_batch_size(
        preparation_params, adjusted_target_samples
    )
    dataset = _configure_final_dataset(dataset, batch_size)

    return dataset, batch_size


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


def _calculate_memory_constraints(
    patch_shape: tuple, dtype: tf.DType, target_samples: int
) -> int:
    """
    Calculate memory-constrained target samples.

    Args:
        patch_shape: Shape of patches
        dtype: Data type of patches
        target_samples: Desired target samples

    Returns:
        int: Memory-constrained target samples
    """
    bytes_per_patch = calculate_bytes_per_patch(patch_shape[1:], dtype)
    safe_memory = 1024**3  # make this configurable
    max_samples_in_memory = safe_memory // bytes_per_patch

    if max_samples_in_memory < target_samples:
        print(
            f"Warning: Reducing target_samples from {target_samples} to {max_samples_in_memory} to fit in GPU memory"
        )
        print(
            f"Memory per patch: {bytes_per_patch / 1024**2:.2f} MB, Safe GPU memory: {safe_memory / 1024**3:.2f} GB"
        )
        return max_samples_in_memory

    return target_samples


def _apply_dataset_repeats(
    dataset: tf.data.Dataset, num_patches: int, adjusted_target_samples: int
) -> tf.data.Dataset:
    """
    Apply repeats to dataset if needed to reach target samples.

    Args:
        dataset: Input dataset
        num_patches: Number of original patches
        adjusted_target_samples: Target number of samples after adjustment

    Returns:
        tf.data.Dataset: Dataset with repeats applied if needed
    """
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


def _upsample_dataset(
    dataset: tf.data.Dataset,
    patch_shape: tuple,
    target_samples: int,
    dtype: tf.DType = None,
) -> tuple:
    """Upsample dataset to reach target number of samples.

    Returns:
        tuple: (upsampled_dataset, adjusted_target_samples)
    """
    num_patches = patch_shape[0]

    # Ensure target_samples is at least equal to the number of available patches
    adjusted_target_samples = max(target_samples, int(num_patches))

    # Apply memory constraints
    adjusted_target_samples = _calculate_memory_constraints(
        patch_shape, dtype, adjusted_target_samples
    )

    # Apply repeats and take to reach target samples
    upsampled_dataset = _apply_dataset_repeats(
        dataset, num_patches, adjusted_target_samples
    )

    return upsampled_dataset, adjusted_target_samples


def create_channel_mask(fieldin_names, noise_channels=None) -> tf.Tensor:
    """
    Create a boolean mask indicating which channels should have noise applied.

    Args:
        fieldin_names: Sequence of field names (e.g., ['thk', 'usurf', 'arrhenius', 'slidingco'] or tuple)
        noise_channels: Sequence of channel names to apply noise to. If None, applies to ('thk', 'usurf')

    Returns:
        tf.Tensor: Boolean mask of shape [num_channels]
    """
    if noise_channels is None:
        noise_channels = ("thk", "usurf")  # Default channels similar to pertubate_X

    mask = [field_name in noise_channels for field_name in fieldin_names]
    return tf.constant(mask, dtype=tf.bool)


@tf.function
def _apply_all_augmentations(
    x,
    rotation_probability,
    flip_probability,
    noise_type,
    noise_scale,
    dtype,
    fieldin_names,
    noise_channels,
):
    rotation_params = RotationParams(probability=rotation_probability)
    flip_params = FlipParams(probability=flip_probability)

    channel_mask = create_channel_mask(fieldin_names, noise_channels)

    noise_params = NoiseParams(
        noise_type=noise_type, noise_scale=noise_scale, channel_mask=channel_mask
    )

    augmentations = [
        RotationAugmentation(rotation_params),
        FlipAugmentation(flip_params),
        NoiseAugmentation(noise_params),
    ]
    for aug in augmentations:
        x = aug.apply(x)
    return tf.cast(x, dtype)
