#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

def cnn(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[1, None, None, nb_inputs])

    conv = inputs

    if cfg.processes.iceflow.emulator.network.activation == "LeakyReLU":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.Activation(cfg.processes.iceflow.emulator.network.activation)

    for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
        conv = tf.keras.layers.Conv3D(
            filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
            kernel_size=(1,cfg.processes.iceflow.emulator.network.conv_ker_size, 
                           cfg.processes.iceflow.emulator.network.conv_ker_size),
            kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
            padding="same",
        )(conv)

        conv = activation(conv)

        if cfg.processes.iceflow.emulator.network.dropout_rate>0:
            conv = tf.keras.layers.Dropout(cfg.processes.iceflow.emulator.network.dropout_rate)(conv)

    for i in range(int(np.log(cfg.processes.iceflow.numerics.Nz)/np.log(2))):
        
        conv = tf.keras.layers.UpSampling3D( size=(2, 1, 1) )(conv)    
            
        conv = tf.keras.layers.Conv3D(
            filters=cfg.processes.iceflow.emulator.network.nb_out_filter/(2**(i+1)),
            kernel_size=(cfg.processes.iceflow.emulator.network.conv_ker_size,
                         cfg.processes.iceflow.emulator.network.conv_ker_size, 
                         cfg.processes.iceflow.emulator.network.conv_ker_size),
            padding="same",
        )(conv)
            
    outputs = conv
 
    outputs = tf.keras.layers.Conv3D(
        filters=nb_outputs,
        kernel_size=(1,1,1),
        kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
        activation=None,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
 
def unet(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    inputs = tf.keras.layers.Input(shape=[1, None, None, nb_inputs])

    conv = inputs[:, 0, :, :, :]

    Nz = cfg.processes.iceflow.numerics.Nz

    from keras_unet_collection import models

    layers = np.arange(int(cfg.processes.iceflow.emulator.network.nb_blocks))

    number_of_filters = [
        cfg.processes.iceflow.emulator.network.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
    ]

    conv = models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs * Nz,
        stack_num_down=2,
        stack_num_up=2,
        activation=cfg.processes.iceflow.emulator.network.activation,
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )(conv)
 
    outputs = tf.transpose( 
                tf.concat([tf.expand_dims(conv[:,:,:,:Nz],axis=1), \
                           tf.expand_dims(conv[:,:,:,Nz:],axis=1)], axis=1)
                          ,perm=[0, 4, 2, 3, 1]
                          )

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

# class FourierLayer(tf.keras.layers.Layer):
#     """Custom layer to perform Fourier Transform."""

#     def __init__(self, **kwargs):
#         super(FourierLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         # Perform the Fourier Transform and cast to complex64
#         return tf.signal.fft2d(tf.cast(inputs, dtype=tf.complex64))

#     def compute_output_shape(self, input_shape):
#         # The output shape is the same as the input shape
#         return input_shape

# def fourier(cfg, nb_inputs, nb_outputs):
#     """
#     Routine to build a convolutional neural network using Fourier layers.
#     """
#     # Define the input layer
#     inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

#     fourier_output = inputs

#     # Determine the activation function based on user parameters
#     if cfg.processes.iceflow.emulator.network.activation == "LeakyReLU":
#         activation = tf.keras.layers.LeakyReLU(alpha=0.01)
#     else:
#         activation = getattr(tf.keras.layers, cfg.processes.iceflow.emulator.network.activation)()

#     # Add Fourier layers, activation, and dropout layers
#     for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
#         # Apply Fourier Layer
#         fourier_output = FourierLayer()(fourier_output)

#         # Separate real and imaginary components
#         real_part = tf.math.real(fourier_output)
#         imag_part = tf.math.imag(fourier_output)

#         # Apply Activation Function to both parts (Real and Imaginary)
#         real_part = activation(real_part)
#         imag_part = activation(imag_part)

#         # Combine back into complex form
#         fourier_output = tf.cast(real_part, dtype=tf.complex64) + 1j * tf.cast(imag_part, dtype=tf.complex64)

#         # Dropout Layer
#         fourier_output = tf.keras.layers.Dropout(cfg.processes.iceflow.emulator.network.dropout_rate)(fourier_output)

#     # Transform back to spatial domain with Inverse FFT
#     spatial_output = tf.signal.ifft2d(fourier_output)

#     # Calculate the magnitude (or you could also use the real part) before the Conv2D layer
#     magnitude_output = tf.abs(spatial_output)  # Use magnitude of the output

#     # Final layer to get the output after applying Conv2D
#     outputs = tf.keras.layers.Conv2D(
#         filters=nb_outputs,
#         kernel_size=(1, 1),
#         kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
#         activation=None,
#     )(magnitude_output)

#     # Create and return the model
#     return tf.keras.models.Model(inputs=inputs, outputs=outputs)
