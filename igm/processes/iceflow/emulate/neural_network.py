#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

def cnn(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    conv = inputs

    if cfg.processes.iceflow.emulator.network.activation == "LeakyReLU":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.Activation(cfg.processes.iceflow.emulator.network.activation)

    for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):
        conv = tf.keras.layers.Conv2D(
            filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
            kernel_size=(cfg.processes.iceflow.emulator.network.conv_ker_size, cfg.processes.iceflow.emulator.network.conv_ker_size),
            kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
            padding="same",
        )(conv)

        conv = activation(conv)

        if cfg.processes.iceflow.emulator.network.dropout_rate>0:
            conv = tf.keras.layers.Dropout(cfg.processes.iceflow.emulator.network.dropout_rate)(conv)
 
    if cfg.processes.iceflow.emulator.network.cnn3d_for_vertical:

        conv = tf.expand_dims(conv, axis=1)

        for i in range(int(np.log(cfg.processes.iceflow.numerics.Nz)/np.log(2))):
                
            conv = tf.keras.layers.Conv3D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter/(2**(i+1)),
                kernel_size=(cfg.processes.iceflow.emulator.network.conv_ker_size,
                            cfg.processes.iceflow.emulator.network.conv_ker_size,
                            cfg.processes.iceflow.emulator.network.conv_ker_size),
                padding="same",
            )(conv)

            conv = tf.keras.layers.UpSampling3D( size=(2, 1, 1) )(conv)   

        conv = tf.transpose( tf.concat([conv[:,:,:,:,0], conv[:,:,:,:,1]], axis=1), perm=[0, 2, 3, 1])
 
    outputs = conv

    outputs = tf.keras.layers.Conv2D(
        filters=nb_outputs,
        kernel_size=(
            1,
            1,
        ),
        kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
        activation=None,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def unet(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    from keras_unet_collection import models

    layers = np.arange(int(cfg.processes.iceflow.emulator.network.nb_blocks))

    number_of_filters = [
        cfg.processes.iceflow.emulator.network.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
    ]

    return models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs,
        stack_num_down=2,
        stack_num_up=2,
        activation=cfg.processes.iceflow.emulator.network.activation,
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )