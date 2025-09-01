import tensorflow as tf
from abc import ABC, abstractmethod


class Emulator(ABC):
    @abstractmethod
    def apply(self, array):
        pass


class LinearRegressor(Emulator):
    def __init__(self, coefficients, b) -> None:
        self.coefficients = coefficients
        self.b = b

    def apply(self, array: tf.Tensor) -> tf.Tensor:
        y = tf.matmul(self.coefficients, array) + self.b

        return y