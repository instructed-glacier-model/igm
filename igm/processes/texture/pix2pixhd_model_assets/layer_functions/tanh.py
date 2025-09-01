import tensorflow.keras as K
from tensorflow.keras import layers


class Tanh(layers.Layer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def call(self, x):
        return K.activations.tanh(x)