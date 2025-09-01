import tensorflow as tf
from abc import ABC, abstractmethod

from .constants import FeatureConstants, ImageConstants

class Normalizer(ABC):
    @abstractmethod
    def normalize(self, image: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def unnormalize(self, image: tf.Tensor) -> tf.Tensor:
        pass




from dataclasses import fields
class FeatureNormalizer(Normalizer):
    def __init__(self, constants: FeatureConstants):
        self.constants = constants
        self.constant_names = [field.name for field in fields(constants)]

    def normalize(self, image: tf.Tensor, variable_name: str) -> tf.Tensor:
        array_max, array_min = getattr(self.constants, variable_name)
        normalized_image = (
            2 * ((image - array_min) / (array_max - array_min)) - 1
        )  # normalize to [-1, 1]

        return normalized_image

    def unnormalize(self, image: tf.Tensor) -> tf.Tensor:
        return image

    def normalize_all(self, image: tf.Tensor) -> tf.Tensor:
        numpy_image = image.numpy()
        for i, feature in enumerate(self.constant_names):
            numpy_image[..., i] = self.normalize(numpy_image[..., i], feature)
            # TODO: make it not cast to numpy and maybe define a custom tf.map or tf.function...

        return tf.convert_to_tensor(numpy_image, dtype=tf.float32)

class ImageNormalizer(Normalizer):
    def __init__(self, constants: ImageConstants):
        self.constants = constants

    def normalize(self, image: tf.Tensor) -> tf.Tensor:
        return image

    def unnormalize(self, image: tf.TensorArray) -> tf.Tensor:
        image = 255 * (image * 0.5 + 0.5)

        return image