import tensorflow as tf
from typing import Any, Dict, Tuple

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .patching import OverlapPatching, GridPatching, Patching
from igm.processes.iceflow.utils import fieldin_to_X_2d, fieldin_to_X_3d


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
    fieldin_names: Tuple[
        str, ...
    ]  # names of input fields for selective noise application
    noise_channels: Tuple[str, ...]  # names of input fields to apply noise


def get_input_params_args(cfg) -> Dict[str, Any]:

    cfg_emulator = cfg.processes.iceflow.emulator

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

    # Determine noise channels - default to ["thk", "usurf"] similar to pertubate_X
    if hasattr(cfg.processes, "data_assimilation"):
        # Use control_list fields that are suitable for noise
        noise_channels = []
        for f in cfg.processes.data_assimilation.control_list:
            if f in ["thk", "usurf"]:  # Only apply noise to these fields
                noise_channels.append(f)
        # Convert to tuple for tf.experimental.ExtensionType compatibility
        noise_channels = tuple(noise_channels) if noise_channels else ("thk", "usurf")
    else:
        # Default noise channels
        noise_channels = ("thk", "usurf")

    return {
        "overlap": 0.25,
        "batch_size": 32,
        "patch_size": 64,
        "rotation_probability": 0.0,
        "flip_probability": 0.0,
        "noise_type": "perlin",
        "noise_scale": 0.1,
        "target_samples": 64,
        "fieldin_names": (
            tuple(cfg.processes.iceflow.emulator.fieldin)
            if hasattr(cfg.processes.iceflow.emulator, "fieldin")
            else ("thk", "usurf", "arrhenius", "slidingco")
        ),
        "noise_channels": noise_channels,
    }


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
        # Single patch case - return input dimensions
        return (
            input_height,
            input_width,
            min(batch_size, max(target_samples, 1)),
        )

    # Calculate overlap parameters using the same logic as OverlapPatching
    height_f = float(input_height)
    width_f = float(input_width)
    patch_size_f = float(patch_size)

    # Calculate minimum stride and number of patches
    min_stride = int(patch_size_f * (1.0 - overlap))

    # Calculate number of patches in each direction
    import math

    n_patches_y = max(
        1, int(math.ceil((height_f - patch_size_f) / float(min_stride))) + 1
    )
    n_patches_x = max(
        1, int(math.ceil((width_f - patch_size_f) / float(min_stride))) + 1
    )

    # Total number of patches
    num_patches = n_patches_y * n_patches_x

    # Patch dimensions are always patch_size x patch_size (except for the edge case above)
    Ny = patch_size
    Nx = patch_size

    # Calculate adjusted target samples (must be at least num_patches)
    adjusted_target_samples = max(target_samples, num_patches)

    # Calculate effective batch size (min of batch_size and adjusted_target_samples)
    effective_batch_size = min(batch_size, adjusted_target_samples)

    return Ny, Nx, effective_batch_size


def split_fieldin_to_patches(cfg, fieldin, patching: OverlapPatching) -> tuple:
    """
    Create base patches without augmentation (static preprocessing).
    This only needs to be done once and can be reused across epochs.

    """

    dim_arrhenius = cfg.processes.iceflow.physics.dim_arrhenius

    if dim_arrhenius == 3:
        X = fieldin_to_X_3d(dim_arrhenius, fieldin)
    elif dim_arrhenius == 2:
        X = fieldin_to_X_2d(fieldin)

    X = tf.squeeze(X, axis=0)  # temporary fix necessary for backwards compatibility

    patches = patching.patch_tensor(X)

    return patches


def create_training_set_from_patches(
    patches: tf.Tensor, preparation_params: PreparationParams
) -> tf.data.Dataset:
    """
    Create a training dataset from pre-computed patches with fresh augmentations.
    This can be called multiple times to get different augmentations.
    """
    patch_shape = tf.shape(patches)
    dtype = patches.dtype

    # Create dataset from patches
    dataset = tf.data.Dataset.from_tensor_slices(patches)

    # Check if augmentations should be applied
    has_rotation = preparation_params.rotation_probability > 0
    has_flip = preparation_params.flip_probability > 0
    has_noise = (
        preparation_params.noise_type != "none" and preparation_params.noise_scale > 0
    )

    apply_augmentations = has_rotation or has_flip or has_noise

    # Apply augmentations only if needed
    if apply_augmentations:
        dataset = dataset.map(
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

    # Shuffle before upsampling
    dataset = dataset.shuffle(buffer_size=100)

    # Check if upsampling is needed and useful
    num_patches = int(patch_shape[0])
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
        should_upsample = False

    # Upsample with repeat and take to reach exactly target_samples
    if should_upsample:
        dataset, adjusted_target_samples = _upsample_dataset(
            dataset,
            patch_shape,
            preparation_params.target_samples,
            dtype,
        )
    else:
        adjusted_target_samples = num_patches

    # if batch size > target_samples, batch_size = target_samples
    batch_size = min(preparation_params.batch_size, adjusted_target_samples)

    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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

    bytes_per_patch = calculate_bytes_per_patch(patch_shape[1:], dtype)

    safe_memory = 1024**3  # make this configurable

    # Calculate maximum samples that fit in memory
    max_samples_in_memory = safe_memory // bytes_per_patch

    if max_samples_in_memory < adjusted_target_samples:
        adjusted_target_samples = max_samples_in_memory
        print(
            f"Warning: Reducing target_samples from {target_samples} to {adjusted_target_samples} to fit in GPU memory"
        )
        print(
            f"Memory per patch: {bytes_per_patch / 1024**2:.2f} MB, Safe GPU memory: {safe_memory / 1024**3:.2f} GB"
        )

    if num_patches < adjusted_target_samples:
        repeats = tf.cast(
            tf.math.ceil(
                tf.cast(adjusted_target_samples, tf.float32)
                / tf.cast(num_patches, tf.float32)
            ),
            tf.int64,
        )
        dataset = dataset.repeat(repeats)
    upsampled_dataset = dataset.take(tf.cast(adjusted_target_samples, tf.int64))
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
