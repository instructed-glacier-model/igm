import tensorflow as tf
from .base import Augmentation


class FlipParams(tf.experimental.ExtensionType):
    probability: float


class FlipAugmentation(Augmentation):
    def __init__(self, params: FlipParams):
        self.params = params

    def apply(self, x):
        # Each flip type is decided independently, allowing for both flips

        if tf.random.uniform([]) < self.params.probability:
            x = tf.image.flip_left_right(x)
        if tf.random.uniform([]) < self.params.probability:
            x = tf.image.flip_up_down(x)
        return x
