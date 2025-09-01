"""
Tensor-based data preprocessing for IGM training.
This module provides tensor-based alternatives to the tf.data.Dataset approach
for better XLA compatibility and consistent compilation behavior.
"""

import tensorflow as tf
from typing import Any, Dict, Tuple

from .augmentations.rotation import RotationAugmentation, RotationParams
from .augmentations.flip import FlipAugmentation, FlipParams
from .augmentations.noise import NoiseAugmentation, NoiseParams
from .data_preprocessing import (
    PreparationParams,
    create_channel_mask,
    _calculate_memory_constraints,
)


def create_training_tensor_from_patches(
    patches: tf.Tensor, preparation_params: PreparationParams
) -> Tuple[tf.Tensor, int]:
    """
    Create a training tensor from pre-computed patches with fresh augmentations.
    Returns a tensor instead of a tf.data.Dataset for consistent XLA compilation.

    Args:
        patches: Pre-computed patches tensor of shape [num_patches, height, width, channels]
        preparation_params: Parameters containing augmentation and batching settings

    Returns:
        tuple: (training_tensor, effective_batch_size)
            - training_tensor: Shape [num_batches, batch_size, height, width, channels]
            - effective_batch_size: Actual batch size used
    """
    patch_shape = tf.shape(patches)
    dtype = patches.dtype
    num_patches = int(patch_shape[0])

    # Start with original patches
    training_samples = patches

    # Apply augmentations if needed
    apply_augmentations = _should_apply_augmentations(preparation_params)

    if apply_augmentations:
        training_samples = _apply_augmentations_to_tensor(
            training_samples, preparation_params, dtype
        )

    # Handle upsampling to reach target_samples
    should_upsample = _should_upsample_tensor(
        preparation_params, num_patches, apply_augmentations
    )

    if should_upsample:
        training_samples, adjusted_target_samples = _upsample_tensor(
            training_samples,
            preparation_params.target_samples,
            dtype,
            preparation_params,
        )
    else:
        adjusted_target_samples = num_patches

    # Calculate effective batch size
    effective_batch_size = _calculate_effective_batch_size(
        preparation_params, adjusted_target_samples
    )

    # Shuffle the samples
    training_samples = tf.random.shuffle(training_samples)

    # Split into batches and reshape to [num_batches, batch_size, height, width, channels]
    training_tensor = _split_tensor_into_batches(training_samples, effective_batch_size)

    return training_tensor, effective_batch_size


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


def _should_upsample_tensor(
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


def _apply_augmentations_to_tensor(
    tensor: tf.Tensor, preparation_params: PreparationParams, dtype: tf.DType
) -> tf.Tensor:
    """
    Apply augmentations to a tensor (similar to _apply_augmentations_to_dataset but for tensors).

    Args:
        tensor: Input tensor of shape [num_samples, height, width, channels]
        preparation_params: Parameters containing augmentation settings
        dtype: Data type for casting

    Returns:
        tf.Tensor: Tensor with augmentations applied
    """

    def apply_augmentations_to_sample(x):
        return _apply_all_augmentations(
            x,
            preparation_params.rotation_probability,
            preparation_params.flip_probability,
            preparation_params.noise_type,
            preparation_params.noise_scale,
            dtype,
            preparation_params.fieldin_names,
            preparation_params.noise_channels,
        )

    # Apply augmentations to each sample using tf.map_fn
    augmented_tensor = tf.map_fn(
        apply_augmentations_to_sample,
        tensor,
        fn_output_signature=tf.TensorSpec(shape=tensor.shape[1:], dtype=dtype),
        parallel_iterations=32,
    )

    return augmented_tensor


def _upsample_tensor(
    tensor: tf.Tensor,
    target_samples: int,
    dtype: tf.DType,
    preparation_params: PreparationParams,
) -> Tuple[tf.Tensor, int]:
    """
    Upsample tensor to reach target number of samples by generating new augmented samples.

    Args:
        tensor: Input tensor of shape [num_samples, height, width, channels]
        target_samples: Target number of samples
        dtype: Data type
        preparation_params: Parameters needed for augmentation

    Returns:
        tuple: (upsampled_tensor, adjusted_target_samples)
    """
    current_shape = tf.shape(tensor)
    num_samples = current_shape[0]

    # Ensure target_samples is at least equal to the number of available samples
    adjusted_target_samples = max(target_samples, int(num_samples))

    # Apply memory constraints
    adjusted_target_samples = _calculate_memory_constraints(
        current_shape, dtype, adjusted_target_samples
    )

    if num_samples >= adjusted_target_samples:
        # Already have enough samples, just take the first adjusted_target_samples
        return tensor[:adjusted_target_samples], adjusted_target_samples

    # Need to generate more samples through augmentation
    additional_samples_needed = adjusted_target_samples - num_samples

    # Create additional samples by repeatedly augmenting the original samples
    augmented_samples = []
    samples_generated = 0

    while samples_generated < additional_samples_needed:
        # How many samples to generate in this round
        samples_this_round = min(
            num_samples, additional_samples_needed - samples_generated
        )

        # Take a subset of original samples (cycling through if needed)
        source_samples = tensor[:samples_this_round]

        # Apply augmentations to create new unique samples
        def apply_augmentations_to_sample(x):
            return _apply_all_augmentations(
                x,
                preparation_params.rotation_probability,
                preparation_params.flip_probability,
                preparation_params.noise_type,
                preparation_params.noise_scale,
                dtype,
                preparation_params.fieldin_names,
                preparation_params.noise_channels,
            )

        # Generate augmented samples
        new_augmented_samples = tf.map_fn(
            apply_augmentations_to_sample,
            source_samples,
            fn_output_signature=tf.TensorSpec(shape=tensor.shape[1:], dtype=dtype),
            parallel_iterations=32,
        )

        augmented_samples.append(new_augmented_samples)
        samples_generated += samples_this_round

    # Combine original samples with all augmented samples
    all_augmented = tf.concat(augmented_samples, axis=0)
    final_tensor = tf.concat([tensor, all_augmented], axis=0)

    return final_tensor, adjusted_target_samples


def _split_tensor_into_batches(tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Split tensor into batches with shape [num_batches, batch_size, height, width, channels].

    Args:
        tensor: Input tensor of shape [total_samples, height, width, channels]
        batch_size: Size of each batch

    Returns:
        tf.Tensor: Batched tensor of shape [num_batches, batch_size, height, width, channels]
    """
    tensor_shape = tf.shape(tensor)
    total_samples = tensor_shape[0]

    # Calculate number of complete batches (drop remainder)
    num_batches = total_samples // batch_size

    # Take only the samples that fit into complete batches
    samples_to_use = num_batches * batch_size
    trimmed_tensor = tensor[:samples_to_use]

    # Reshape to [num_batches, batch_size, height, width, channels]
    height, width, channels = tensor_shape[1], tensor_shape[2], tensor_shape[3]
    batched_tensor = tf.reshape(
        trimmed_tensor, [num_batches, batch_size, height, width, channels]
    )

    return batched_tensor


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
    """
    Apply all augmentations to a single sample.

    Args:
        x: Input tensor sample
        rotation_probability: Probability of rotation
        flip_probability: Probability of flipping
        noise_type: Type of noise to apply
        noise_scale: Scale of noise
        dtype: Data type for casting
        fieldin_names: Names of input fields
        noise_channels: Channels to apply noise to

    Returns:
        tf.Tensor: Augmented sample
    """
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


def ensure_fixed_tensor_shape(
    tensor: tf.Tensor, expected_shape: Tuple[int, int, int]
) -> tf.Tensor:
    """
    Ensure tensor has a fixed shape for consistent XLA compilation.

    Args:
        tensor: Input tensor
        expected_shape: Expected shape tuple (height, width, channels)

    Returns:
        tf.Tensor: Tensor with enforced shape
    """
    return tf.ensure_shape(tensor, expected_shape)
