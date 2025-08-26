import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple
from typeguard import typechecked


class Patching(ABC):
    """
    Abstract base class for tensor patching strategies.

    Patching takes an input tensor with shape (height, width, channels)
    and splits it into smaller patches for processing. Different strategies
    can be used for determining patch layout, overlap, and stacking behavior.

    The framework supports:
    - patch_tensor: main method that splits tensor into patches
    - Configurable patch size and overlap strategies
    - All patches stacked along the batch dimension for efficient processing
    """

    def __init__(self, patch_size: int, memory_checker=None):
        """
        Initialize base patching.

        Args:
            patch_size: Size of each patch (height and width).
        """
        # Import here to avoid circular imports
        self.patch_size = patch_size

    @abstractmethod
    @typechecked
    def patch_tensor(self, X: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Split input tensor into patches.

        This method must be implemented by subclasses to define the specific
        patching strategy (e.g., with/without overlap, different stacking methods).

        Args:
            X: Input tensor of shape (height, width, channels).
            **kwargs: Additional parameters specific to each implementation.

        Returns:
            Tensor containing patches stacked along the batch dimension.
        """
        pass

    @tf.function(reduce_retracing=True)
    def _validate_input(self, X: tf.Tensor) -> None:
        """
        Validate input tensor shape and properties.

        Args:
            X: Input tensor to validate.

        Raises:
            tf.errors.InvalidArgumentError: If input is invalid.
        """
        tf.debugging.assert_rank(
            X, 3, "Input tensor must be 3D (height, width, channels)"
        )
        tf.debugging.assert_greater(tf.shape(X)[0], 0, "Height must be positive")
        tf.debugging.assert_greater(tf.shape(X)[1], 0, "Width must be positive")
        tf.debugging.assert_greater(tf.shape(X)[2], 0, "Channels must be positive")

    @tf.function(reduce_retracing=True)
    def _get_patch_dimensions(self, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get height and width from input tensor.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tuple of (height, width) as TensorFlow tensors.
        """
        shape = tf.shape(X)
        return shape[0], shape[1]

    @tf.function(reduce_retracing=True)
    def _extract_patch(
        self, X: tf.Tensor, start_y: tf.Tensor, start_x: tf.Tensor
    ) -> tf.Tensor:
        """
        Extract a single patch from the input tensor.

        Args:
            X: Input tensor of shape (height, width, channels).
            start_y: Starting y coordinate for the patch.
            start_x: Starting x coordinate for the patch.

        Returns:
            Extracted patch of shape (patch_size, patch_size, channels).
        """
        return X[
            start_y : start_y + self.patch_size,
            start_x : start_x + self.patch_size,
            :,
        ]


class OverlapPatching(Patching):
    """
    Patching strategy with configurable overlap between patches.

    This implementation splits the input tensor into patches with a specified
    minimum overlap. The actual overlap may be slightly larger to ensure
    complete coverage of the input tensor.
    """

    def __init__(self, patch_size: int, overlap: float = 0.25, memory_checker=None):
        """
        Initialize overlap patching.

        Args:
            patch_size: Size of each patch (height and width).
            overlap: Minimum fractional overlap between patches (0.0 to 1.0).
        """
        super().__init__(patch_size, memory_checker)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Overlap must be in range [0.0, 1.0)")
        self.overlap = overlap

    @tf.function(reduce_retracing=True)
    def _calculate_overlap_parameters(
        self, height: tf.Tensor, width: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate overlap patching parameters.

        Args:
            height: Height of input tensor.
            width: Width of input tensor.

        Returns:
            Tuple of (n_patches_y, n_patches_x, stride_y, stride_x).
        """
        # Return single patch if input is smaller than patch size
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        patch_size_f = tf.cast(self.patch_size, tf.float32)

        # Calculate minimum stride and number of patches
        min_stride = tf.cast(patch_size_f * (1.0 - self.overlap), tf.int32)

        n_patches_y = tf.maximum(
            1,
            tf.cast(
                tf.math.ceil(
                    (height_f - patch_size_f) / tf.cast(min_stride, tf.float32)
                ),
                tf.int32,
            )
            + 1,
        )
        n_patches_x = tf.maximum(
            1,
            tf.cast(
                tf.math.ceil(
                    (width_f - patch_size_f) / tf.cast(min_stride, tf.float32)
                ),
                tf.int32,
            )
            + 1,
        )

        # Calculate actual strides for exact coverage
        stride_y = tf.cond(
            n_patches_y > 1,
            lambda: tf.cast(
                (height_f - patch_size_f) / tf.cast(n_patches_y - 1, tf.float32),
                tf.int32,
            ),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        stride_x = tf.cond(
            n_patches_x > 1,
            lambda: tf.cast(
                (width_f - patch_size_f) / tf.cast(n_patches_x - 1, tf.float32),
                tf.int32,
            ),
            lambda: tf.constant(0, dtype=tf.int32),
        )

        return n_patches_y, n_patches_x, stride_y, stride_x

    @tf.function(reduce_retracing=True)
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Split input tensor into overlapping patches.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tensor of patches with shape (num_patches, patch_size, patch_size, channels).
            All patches are stacked along the batch dimension.
        """
        self._validate_input(X)
        height, width = self._get_patch_dimensions(X)

        # Handle case where input is smaller than patch size
        if tf.logical_and(self.patch_size > width, self.patch_size > height):
            # Add batch dimension and return
            return tf.expand_dims(X, axis=0)

        n_patches_y, n_patches_x, stride_y, stride_x = (
            self._calculate_overlap_parameters(height, width)
        )

        # Generate patch coordinates
        y_coords = tf.range(n_patches_y) * stride_y
        x_coords = tf.range(n_patches_x) * stride_x

        # Create mesh grid of coordinates
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")
        y_flat = tf.reshape(y_grid, [-1])
        x_flat = tf.reshape(x_grid, [-1])

        # Extract all patches using tf.map_fn for efficiency
        patches = tf.map_fn(
            lambda coords: self._extract_patch(X, coords[0], coords[1]),
            tf.stack([y_flat, x_flat], axis=1),
            fn_output_signature=tf.TensorSpec(
                shape=[self.patch_size, self.patch_size, None], dtype=X.dtype
            ),
            parallel_iterations=10,
        )

        # patches shape: (num_patches, patch_size, patch_size, channels)
        return patches


class GridPatching(Patching):
    """
    Patching strategy that divides input into a regular grid without overlap.

    This implementation splits the input tensor into non-overlapping patches
    arranged in a regular grid. All patches are stacked along the batch dimension.
    """

    def __init__(self, patch_size: int, memory_checker=None):
        """
        Initialize grid patching.

        Args:
            patch_size: Size of each patch (height and width).
        """
        super().__init__(patch_size, memory_checker)

    @tf.function(reduce_retracing=True)
    def _calculate_grid_parameters(
        self, height: tf.Tensor, width: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate grid patching parameters.

        Args:
            height: Height of input tensor.
            width: Width of input tensor.

        Returns:
            Tuple of (n_patches_y, n_patches_x, patch_height, patch_width).
        """
        # Calculate number of patches in each dimension
        n_patches_y = height // self.patch_size + 1
        n_patches_x = width // self.patch_size + 1

        # Calculate actual patch dimensions (may be smaller than patch_size)
        patch_height = height // n_patches_y
        patch_width = width // n_patches_x

        return n_patches_y, n_patches_x, patch_height, patch_width

    @tf.function(reduce_retracing=True)
    def _extract_grid_patch(
        self,
        X: tf.Tensor,
        i: tf.Tensor,
        j: tf.Tensor,
        patch_height: tf.Tensor,
        patch_width: tf.Tensor,
    ) -> tf.Tensor:
        """
        Extract a patch from the regular grid.

        Args:
            X: Input tensor of shape (height, width, channels).
            i: Grid row index.
            j: Grid column index.
            patch_height: Height of each patch.
            patch_width: Width of each patch.

        Returns:
            Extracted patch.
        """
        start_y = j * patch_height
        start_x = i * patch_width
        end_y = start_y + patch_height
        end_x = start_x + patch_width

        return X[start_y:end_y, start_x:end_x, :]

    @tf.function(reduce_retracing=True)
    def _extract_all_grid_patches(
        self,
        X: tf.Tensor,
        n_patches_y: tf.Tensor,
        n_patches_x: tf.Tensor,
        patch_height: tf.Tensor,
        patch_width: tf.Tensor,
    ) -> tf.Tensor:
        """
        Extract all grid patches using TensorFlow operations.

        Args:
            X: Input tensor.
            n_patches_y: Number of patches in y direction.
            n_patches_x: Number of patches in x direction.
            patch_height: Height of each patch.
            patch_width: Width of each patch.

        Returns:
            All patches stacked together.
        """
        # Create coordinate meshes
        i_range = tf.range(n_patches_x)
        j_range = tf.range(n_patches_y)

        # Create all coordinate combinations
        i_grid, j_grid = tf.meshgrid(i_range, j_range, indexing="ij")
        i_flat = tf.reshape(i_grid, [-1])
        j_flat = tf.reshape(j_grid, [-1])

        # Extract patches using map_fn
        coordinates = tf.stack([i_flat, j_flat], axis=1)

        def extract_single_patch(coords):
            i, j = coords[0], coords[1]
            return self._extract_grid_patch(X, i, j, patch_height, patch_width)

        patches = tf.map_fn(
            extract_single_patch,
            coordinates,
            fn_output_signature=tf.TensorSpec(shape=[None, None, None], dtype=X.dtype),
            parallel_iterations=1,  # Set to 1 to avoid warning in eager execution
        )

        return patches

    @typechecked
    def patch_tensor(self, X: tf.Tensor) -> tf.Tensor:
        """
        Split input tensor into grid patches.

        Args:
            X: Input tensor of shape (height, width, channels).

        Returns:
            Tensor of patches with shape (num_patches, patch_height, patch_width, channels).
            All patches are stacked along the batch dimension.
        """
        self._validate_input(X)
        height, width = self._get_patch_dimensions(X)

        n_patches_y, n_patches_x, patch_height, patch_width = (
            self._calculate_grid_parameters(height, width)
        )

        # Extract all patches
        patches = self._extract_all_grid_patches(
            X, n_patches_y, n_patches_x, patch_height, patch_width
        )

        # patches shape: (num_patches, patch_height, patch_width, channels)
        return patches
