import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

weight_init = {}
weight_init['conv'] = tf.random_normal_initializer(0.0, 0.02)
weight_init['bn_gamma'] = tf.random_normal_initializer(1.0, 0.02)
weight_init['bn_beta'] = tf.zeros_initializer()

class ResnetBlock(layers.Layer):
    def __init__(self, dim, padding_type, norm_layer,
                 activation=layers.ReLU(), use_dropout=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        # CONSTANT REFLECT SYMMETRIC
        model = K.Sequential((layers.ZeroPadding2D(padding=(1, 1)),
                              layers.Conv2D(dim, 3, kernel_initializer=weight_init['conv']),
                              norm_layer(groups=-1),
                              activation
                              ))
        if use_dropout:
            model.add(layers.Dropout(0.5))
        model.add(layers.ZeroPadding2D([1, 1]))
        model.add(layers.Conv2D(dim, 3, kernel_initializer=weight_init['conv']))
        model.add(norm_layer(groups=-1))
        self.model = model

    def call(self, x):
        identity = x
        x = self.model(x)
        out = x + identity

        return out