import tensorflow as tf
from tensorflow.keras import layers


class ReflectionPad2d(layers.Layer):
    def __init__(self, paddings=(1, 1), **kwargs):
        self.paddings = tuple(paddings)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPad2d, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1]+self.paddings[0], s[2]+self.paddings[1], s[3])

    def call(self, x):
        w_pad, h_pad = self.paddings
        x = tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')
        return x