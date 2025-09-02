# Tensor-based data preprocessing for IGM training.
# This module provides tensor-based alternatives to the tf.data.Dataset approach
# for better XLA compatibility and consistent compilation behavior.

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

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

    IMPORTANT: Augmentations are applied at most once per sample.
    If augmentations are enabled, we upsample first (by replication only),
    then apply a single augmentation pass to the entire pool.

    Args:
        patches: Pre-computed patches tensor of shape [num_patches, height, width, channels]
        preparation_params: Parameters containing augmentation and batching settings

    Returns:
        tuple: (training_tensor, effective_batch_size)
            - training_tensor: Shape [num_batches, batch_size, height, width, channels]
            - effective_batch_size: Actual batch size used (Python int)
    """
    patch_shape = tf.shape(patches)
    dtype = patches.dtype
    num_patches_t = patch_shape[0]

    training_samples = patches

    # Decide whether to upsample (pure-Python decision)
    apply_augmentations = _should_apply_augmentations(preparation_params)
    should_upsample = _should_upsample_tensor(
        preparation_params=preparation_params,
        num_patches=_to_py_int(num_patches_t),
        apply_augmentations=apply_augmentations,
    )

    # If we need more samples and we will augment, replicate first (no augmentation here)
    if should_upsample:
        training_samples, adjusted_target_samples = _upsample_tensor(
            training_samples,
            preparation_params.target_samples,
            dtype,
            preparation_params,
        )
    else:
        adjusted_target_samples = _to_py_int(num_patches_t)

    # Apply exactly one augmentation pass if enabled (now over the full pool)
    if apply_augmentations:
        training_samples = _apply_augmentations_to_tensor(
            training_samples, preparation_params, dtype
        )

    # Enforce static inner shape to keep XLA happy
    if patches.shape.rank == 4 and all(d is not None for d in patches.shape[1:]):
        training_samples = ensure_fixed_tensor_shape(
            training_samples, (patches.shape[1], patches.shape[2], patches.shape[3])
        )

    # Compute effective batch size as Python int (preserve original API)
    effective_batch_size = _calculate_effective_batch_size(
        preparation_params, adjusted_target_samples
    )

    # Shuffle (non-deterministic version, per your request)
    training_samples = tf.random.shuffle(training_samples)

    # Batch into [num_batches, batch_size, H, W, C]
    training_tensor = _split_tensor_into_batches(training_samples, effective_batch_size)

    return training_tensor, effective_batch_size


def _should_apply_augmentations(preparation_params: PreparationParams) -> bool:
    """
    Determine if any augmentations should be applied based on parameters.
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
    Pure-Python decision to keep tracing out of the graph.
    Upsample only if target_samples > num_patches AND augmentations are enabled.
    (Otherwise we'd create identical copies, which we avoid.)
    """
    if preparation_params.target_samples <= num_patches:
        return False
    if not apply_augmentations:
        # Avoid duplicating identical samples without augmentations.
        return False
    return True


def _calculate_effective_batch_size(
    preparation_params: PreparationParams, adjusted_target_samples: int
) -> int:
    """
    Calculate the effective batch size based on parameters and available samples.
    Returns a Python int to preserve original behavior.
    """
    return min(preparation_params.batch_size, adjusted_target_samples)


@tf.function
def _apply_augmentations_to_tensor(
    tensor: tf.Tensor, preparation_params: PreparationParams, dtype: tf.DType
) -> tf.Tensor:
    """
    Apply augmentations to a tensor in a compiled graph.
    Augmentation objects and masks are created once per call (per-tensor),
    not per-sample.

    """
    # Precompute channel mask and augmentation objects once
    channel_mask = create_channel_mask(
        preparation_params.fieldin_names, preparation_params.noise_channels
    )

    rotation = RotationAugmentation(
        RotationParams(probability=preparation_params.rotation_probability)
    )
    flip = FlipAugmentation(FlipParams(probability=preparation_params.flip_probability))
    noise = NoiseAugmentation(
        NoiseParams(
            noise_type=preparation_params.noise_type,
            noise_scale=preparation_params.noise_scale,
            channel_mask=channel_mask,
        )
    )

    def apply_all(x):
        x = rotation.apply(x)
        x = flip.apply(x)
        x = noise.apply(x)
        return tf.cast(x, dtype)

    # Vectorize across the leading (sample) dimension
    return tf.vectorized_map(apply_all, tensor)


def _upsample_tensor(
    tensor: tf.Tensor,
    target_samples: int,
    dtype: tf.DType,
    preparation_params: PreparationParams,
) -> Tuple[tf.Tensor, int]:
    """
    Upsample tensor to reach target number of samples by generating replicated samples.
    NOTE: No augmentation is performed here. Augmentation is applied exactly once
    later to the whole pool if enabled.

    Returns:
        (final_tensor, adjusted_target_samples:int)
    """
    current_shape = tf.shape(tensor)
    num_samples_t = current_shape[0]

    # Ensure target not below existing count
    adjusted_target_samples_t = tf.maximum(
        tf.cast(target_samples, num_samples_t.dtype), num_samples_t
    )

    # Respect memory constraints (may return Tensor or Python int)
    adjusted_target_samples_mc = _calculate_memory_constraints(
        current_shape, dtype, adjusted_target_samples_t
    )

    # Normalize to a Tensor for internal math
    adjusted_target_samples_t = (
        tf.convert_to_tensor(adjusted_target_samples_mc, dtype=num_samples_t.dtype)
        if not isinstance(adjusted_target_samples_mc, tf.Tensor)
        else tf.cast(adjusted_target_samples_mc, num_samples_t.dtype)
    )

    # If we already have enough, truncate and return
    if _to_py_int(num_samples_t) >= _to_py_int(adjusted_target_samples_t):
        adjusted_py = _to_py_int(adjusted_target_samples_t)
        return tensor[:adjusted_py], adjusted_py

    # Extras needed
    extras_needed_t = adjusted_target_samples_t - num_samples_t

    # Replicate originals once to cover extras, slice exact count
    # reps = ceil(extras_needed / num_samples)
    reps_t = tf.math.floordiv(extras_needed_t + num_samples_t - 1, num_samples_t)
    replicated = tf.tile(tensor, [reps_t + 1, 1, 1, 1])[:extras_needed_t]

    final_tensor = tf.concat([tensor, replicated], axis=0)
    adjusted_py = _to_py_int(adjusted_target_samples_t)
    return final_tensor, adjusted_py


@tf.function
def _split_tensor_into_batches(tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Split tensor into batches with shape [num_batches, batch_size, height, width, channels].
    Drops the remainder to keep full batches only.
    """
    tf.debugging.assert_greater(batch_size, 0, message="batch_size must be > 0")

    total = tf.shape(tensor)[0]
    num_batches = total // batch_size
    samples_to_use = num_batches * batch_size
    trimmed = tensor[:samples_to_use]

    shape_rest = tf.shape(trimmed)[1:]  # [H, W, C]
    h, w, c = shape_rest[0], shape_rest[1], shape_rest[2]
    return tf.reshape(trimmed, [num_batches, batch_size, h, w, c])


@tf.function
def ensure_fixed_tensor_shape(
    tensor: tf.Tensor, expected_shape: Tuple[int, int, int]
) -> tf.Tensor:
    """
    Ensure tensor has a fixed inner shape [H, W, C] for consistent XLA compilation.
    """
    return tf.ensure_shape(
        tensor, [None, expected_shape[0], expected_shape[1], expected_shape[2]]
    )


# ----------------
# Small utilities
# ----------------


def _to_py_int(x) -> int:
    """Convert a scalar Tensor or Python numeric to a Python int (eager-safe)."""
    if isinstance(x, tf.Tensor):
        return int(x.numpy())
    return int(x)
