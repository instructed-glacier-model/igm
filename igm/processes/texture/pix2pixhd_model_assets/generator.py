import tensorflow as tf
import tensorflow.keras as K
# from tensorflow_addons.layers import InstanceNormalization # depreciated in tf 2.14
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras import layers
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Input

from .layer_functions.reflection_pad2d import ReflectionPad2d
from .layer_functions.resnet_block import ResnetBlock
from .layer_functions.tanh import Tanh
from .layer_functions.bilinear_upsampling import BilinearUpSampling

weight_init = {}
weight_init['conv'] = tf.random_normal_initializer(0.0, 0.02)
weight_init['bn_gamma'] = tf.random_normal_initializer(1.0, 0.02)
weight_init['bn_beta'] = tf.zeros_initializer()


class LocalEnhancer(K.Model):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=GroupNormalization, padding_type='REFLECT'):
        super(LocalEnhancer, self).__init__()
        
        # Setup variables
        self.n_local_enhancers = n_local_enhancers
        paddings = (3, 3)
        activation = layers.LeakyReLU(0.2)
        # self.split_model = split_model
        
        # Setup Global Generator to use
        ngf_global = ngf * (2**n_local_enhancers)

        model_global = GlobalGenerator(input_nc, output_nc, ngf=ngf_global, n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer).layers[0].layers

        model_global = model_global[:-3] # get rid of final convolution layers
        self.model = K.Sequential(model_global)

        # # Single Local Enhancer
        # Downsampling layers
        ngf_global = ngf * (2**(n_local_enhancers-1))
        
        model_downsample = K.Sequential(name="DownSampler")
        model_downsample.add(ReflectionPad2d(paddings))
        model_downsample.add(layers.Conv2D(ngf_global, 7, kernel_initializer=weight_init['conv']))
        model_downsample.add(norm_layer(groups=-1)) # can't seem to figure out how to alter RandomNormal mean/std
        model_downsample.add(activation)
        model_downsample.add(layers.Conv2D(ngf_global * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv']))
        model_downsample.add(norm_layer(groups=-1)) # can't seem to figure out how to alter RandomNormal mean/std
        model_downsample.add(activation)

        # Residual blocks
        model_upsample = K.Sequential(name="ResnetAndUpsampler")
        for i in range(n_blocks_local):
            model_upsample.add(ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer))

        # Upsampling layers
        # model_upsample.add(layers.Conv2DTranspose(ngf_global, 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv'])) # change this to bilinear?
        model_upsample.add(BilinearUpSampling(ngf=ngf_global, mult=2, weight_init=weight_init, name=f"UpConv2D_0")) # FIX THIS TO WORK WITH LOCAL
        model_upsample.add(norm_layer(groups=-1)) # can't seem to figure out how to alter RandomNormal mean/std
        model_upsample.add(activation)

        # Final Convolutional
        model_upsample.add(ReflectionPad2d(paddings))
        model_upsample.add(layers.Conv2D(output_nc, 7, kernel_initializer=weight_init['conv'])) # change this to bilinear?
        model_upsample.add(Tanh(name='LocalOutput'))

        # Create model attributes for calling function
        self.model_downsampler = model_downsample
        self.model_upsampler = model_upsample

        # Layer to reduce resolution by half...
        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def call(self, x):

        # Initial input (full resolution image)
        input = [x]
        
        # Creating low resolution input -> input is now [high_res, low_res]
        low_res_input = self.downsample(input[-1])
        input.append(low_res_input)

        # Pass low resolution input into global model to get feature map
        global_feature_map = self.model(input[-1])

        # Get downsampling and upsampling parts of the local enhancer
        model_downsample = getattr(self, 'model_downsampler')
        model_upsample = getattr(self, 'model_upsampler')
        
        # Get high resolution input and pass it through the local enhancers downsampler
        output = model_upsample(model_downsample(input[0]) + global_feature_map)

        return output
    
class GlobalGenerator(K.Model):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=GroupNormalization,
                 padding_type='REFLECT'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        
        # Setup initial variables
        self.padding_type = padding_type
        self.output_nc = output_nc
        paddings = (3, 3)

        # Setup Global Model
        model = K.Sequential(name="GlobalGenerator")
        model.add(layers.InputLayer([None, None, input_nc], name=f"Input"))
        
        # Initial Block
        model.add(ReflectionPad2d(paddings, name=f"InReflectPad"))
        model.add(layers.Conv2D(ngf, 7, kernel_initializer=weight_init['conv'], name=f"InConv2D"))
        model.add(norm_layer(name=f"InNorm", groups=-1))
        model.add(layers.LeakyReLU(0.2, name=f"InActivation"))

        # Downsampling blocks
        for i in range(n_downsampling):
            mult = 2**i
            # model.add(ReflectionPad2d(paddings, name=f"ReflectPad_{i}"))
            model.add(layers.Conv2D(ngf * mult * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv'], name=f"Conv2D_{i}"))
            model.add(norm_layer(name=f"DownNorm_{i}", groups=-1))
            model.add(layers.LeakyReLU(0.2, name=f"DownActivation_{i}"))

        # Resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model.add(ResnetBlock(dim=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=layers.ReLU(), name=f"Resnet_{i}"))

        # Upsampling blocks
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model.add(layers.Conv2DTranspose(int(ngf * mult / 2), 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv'], name=f"TransConv2D_{i}"))
            model.add(BilinearUpSampling(ngf=ngf, mult=mult, weight_init=weight_init))
            model.add(norm_layer(name=f"UpNorm_{i}", groups=-1))
            model.add(layers.LeakyReLU(0.2, name=f"UpActivation_{i}"))
            
        # Final layers
        model.add(ReflectionPad2d(paddings, name=f"OutReflectPad"))
        model.add(layers.Conv2D(self.output_nc, 7, kernel_initializer=weight_init['conv'], name=f"OutConv2D"))
        model.add(Tanh(name='GlobalOutput'))

        # Save global model
        self.model = model

    # @tf.function
    def call(self, x):
        x = self.model(x)

        return x