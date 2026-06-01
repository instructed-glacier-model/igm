# tests/test_iceflow/unified/optimizers/vector_mapping.py

#!/usr/bin/env python3
import tensorflow as tf
from typing import List, Tuple
from igm.processes.iceflow.unified.mappings.mapping import Mapping


class VectorMapping(Mapping):
    """
    Minimal Mapping for optimizer tests.
    - Parameters: a single trainable vector theta ∈ R^n
    - U,V: dummy tensors (cost_fn will ignore them and read theta directly)
    """

    def __init__(self, n: int, dtype=tf.float32):
        super().__init__(bcs=[])
        self.theta = tf.Variable(
            tf.zeros([n], dtype=dtype), trainable=True, name="theta"
        )
        self._shapes: List[tf.TensorShape] = [self.theta.shape]
        self._sizes: List[tf.Tensor] = [tf.size(self.theta)]

    # --- Forward (dummy) ---
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Produce small dummy tensors; shapes are irrelevant for the test cost_fn.
        z = tf.zeros([1, 1, 1, 1], dtype=self.theta.dtype)
        return z, z

    # --- Parameter plumbing ---
    def get_theta(self) -> List[tf.Variable]:
        return [self.theta]

    def set_theta(self, theta: List[tf.Tensor]) -> None:
        if len(theta) != 1:
            raise ValueError("TestVectorMapping.set_theta: expected a single tensor.")
        self.theta.assign(theta[0])

    def copy_theta(self, theta: List[tf.Variable]) -> List[tf.Tensor]:
        return [theta[0].read_value()]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def flatten_theta(self, theta: List[tf.Variable | tf.Tensor]) -> tf.Tensor:
        return tf.reshape(theta[0], [-1])

    def unflatten_theta(self, theta_flat: tf.Tensor) -> List[tf.Tensor]:
        return [tf.reshape(theta_flat, self.theta.shape)]

    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs

    # --- Halt criterion (none) ---
    def check_halt_criterion(self, iteration: int, cost: tf.Tensor):
        # Keep running unless Optimizer stops by its own criteria
        return tf.constant(False, tf.bool), tf.constant("", tf.string)


# --- Bounded mapping for box-constraint tests --------------------------------


class BoundedVectorMapping(VectorMapping):
    """
    Extends the simple vector mapping with θ-space box bounds to activate
    the projected path line-search and free-mask logic in optimizers.
    """

    def __init__(self, n: int, L: float, U: float, dtype=tf.float64):
        super().__init__(n=n, dtype=dtype)
        L = tf.convert_to_tensor(L, dtype=self.theta.dtype)
        U = tf.convert_to_tensor(U, dtype=self.theta.dtype)
        self._L = tf.fill([n], L)
        self._U = tf.fill([n], U)

    # This method is probed via hasattr(map, "get_box_bounds_flat")
    def get_box_bounds_flat(self):
        return self._L, self._U


# --- Simple identity sampler for tests ---------------------------------------


class IdentitySampler:
    """
    Mock sampler for optimizer tests.
    Returns inputs wrapped in batch dimension [1, ...] to match expected
    shape [M, B, H, W, C] where M=1 (single batch).
    """

    def __init__(self):
        self.dynamic_augmentation = False

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs is expected to be [B, H, W, C] or similar
        # Return [1, B, H, W, C] to indicate single batch
        return tf.expand_dims(inputs, axis=0)
