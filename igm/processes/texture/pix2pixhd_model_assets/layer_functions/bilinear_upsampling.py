from tensorflow.keras import layers

class BilinearUpSampling(layers.Layer):
    def __init__(self, ngf, mult, weight_init, **kwargs):
        super(BilinearUpSampling, self).__init__(**kwargs)
        self.ngf = ngf
        self.mult = mult
        self.weight_init = weight_init
        # self.name = name

    def build(self, input_shape):
        self.upsampling = layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv = layers.Conv2D(filters=int(self.ngf * self.mult / 2), kernel_size=3, padding='same', kernel_initializer=self.weight_init['conv'])

    def call(self, inputs):
        x = self.upsampling(inputs)
        x = self.conv(x)
        return x