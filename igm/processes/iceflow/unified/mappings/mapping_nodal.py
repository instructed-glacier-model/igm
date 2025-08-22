import tensorflow as tf

from .mapping import Mapping


class MappingNodal(Mapping):
    def __init__(self, u_guess: tf.Tensor):
        self.shape = u_guess.shape
        self.type = u_guess.dtype
        self.u = tf.Variable(u_guess, trainable=True)

    def get_u_impl(self) -> tf.Variable:
        return self.u

    def copy_w(self, w: list[tf.Variable]) -> list[tf.Tensor]:
        return [w[0].read_value()]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def get_w(self) -> list[tf.Variable]:
        return [self.u]

    def set_w(self, w: list[tf.Tensor]) -> None:
        self.u.assign(w[0])

    def flatten_w(self, w: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        return tf.reshape(w[0], [-1])

    def unflatten_w(self, w_flat) -> list[tf.Tensor]:
        return [tf.reshape(w_flat, self.shape)]
