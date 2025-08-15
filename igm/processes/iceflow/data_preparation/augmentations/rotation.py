import tensorflow as tf
from .base import Augmentation


class RotationParams(tf.experimental.ExtensionType):
    probability: float


class RotationAugmentation(Augmentation):
    def __init__(self, params: RotationParams):
        self.params = params

    def apply(self, x):
        # Check if image is square - only apply rotation to square images
        height = tf.shape(x)[0]
        width = tf.shape(x)[1]
        is_square = tf.equal(height, width)

        # Only rotate if probability check passes AND image is square
        should_rotate = tf.logical_and(
            tf.random.uniform([]) < self.params.probability, is_square
        )

        if should_rotate:
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            x = tf.image.rot90(x, k)
        return x
