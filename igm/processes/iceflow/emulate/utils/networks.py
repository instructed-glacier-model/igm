#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf


def cnn(cfg, nb_inputs, nb_outputs, input_normalizer=None):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    if input_normalizer is not None:
        inputs = input_normalizer(inputs)

    use_batch_norm = (
        hasattr(cfg.processes.iceflow.emulator.network, "batch_norm")
        and cfg.processes.iceflow.emulator.network.batch_norm
    )

    use_residual = (
        hasattr(cfg.processes.iceflow.emulator.network, "residual")
        and cfg.processes.iceflow.emulator.network.residual
    )

    use_separable = (
        hasattr(cfg.processes.iceflow.emulator.network, "separable")
        and cfg.processes.iceflow.emulator.network.separable
    )

    if hasattr(cfg.processes.iceflow.emulator.network, "l2_reg"):
        kernel_regularizer = tf.keras.regularizers.l2(
            cfg.processes.iceflow.emulator.network.l2_reg
        )
    else:
        kernel_regularizer = None

    conv = inputs

    if cfg.processes.iceflow.emulator.network.activation.lower() == "leakyrelu":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.Activation(
            cfg.processes.iceflow.emulator.network.activation
        )

    for i in range(int(cfg.processes.iceflow.emulator.network.nb_layers)):

        residual_in = conv

        if use_separable:
            conv = tf.keras.layers.SeparableConv2D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
                kernel_size=(cfg.processes.iceflow.emulator.network.conv_ker_size,) * 2,
                depthwise_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                pointwise_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                padding="same",
                depthwise_regularizer=kernel_regularizer,
                pointwise_regularizer=kernel_regularizer,
            )(conv)

        else:
            conv = tf.keras.layers.Conv2D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter,
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                kernel_initializer=cfg.processes.iceflow.emulator.network.weight_initialization,
                padding="same",
                kernel_regularizer=kernel_regularizer,
            )(conv)

        if use_batch_norm:
            conv = tf.keras.layers.BatchNormalization()(conv)

        conv = activation(conv)

        if cfg.processes.iceflow.emulator.network.dropout_rate > 0:
            conv = tf.keras.layers.Dropout(
                cfg.processes.iceflow.emulator.network.dropout_rate
            )(conv)

        if use_residual and i % 2 == 1 and conv.shape[-1] == residual_in.shape[-1]:
            conv = tf.keras.layers.Add()([conv, residual_in])

    if cfg.processes.iceflow.emulator.network.cnn3d_for_vertical:

        conv = tf.expand_dims(conv, axis=1)

        for i in range(int(np.log(cfg.processes.iceflow.numerics.Nz) / np.log(2))):

            conv = tf.keras.layers.Conv3D(
                filters=cfg.processes.iceflow.emulator.network.nb_out_filter
                / (2 ** (i + 1)),
                kernel_size=(
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                    cfg.processes.iceflow.emulator.network.conv_ker_size,
                ),
                padding="same",
            )(conv)

            conv = tf.keras.layers.UpSampling3D(size=(2, 1, 1))(conv)

        conv = tf.transpose(
            tf.concat([conv[:, :, :, :, 0], conv[:, :, :, :, 1]], axis=1),
            perm=[0, 2, 3, 1],
        )

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
        cfg.processes.iceflow.emulator.network.nb_out_filter * 2 ** (layers[i])
        for i in range(len(layers))
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


def build_norm_layer(cfg, nb_inputs, scales):

    # check that scales has length nb_inputs
    if len(scales) != nb_inputs:
        raise ValueError(
            f"Expected inputs scales of length {nb_inputs}, got {len(scales)}"
        )

    norm = tf.keras.layers.Normalization(axis=-1)  # per-channel (last dim)
    norm.build((None, None, None, nb_inputs))  # N,H,W,C so variables exist

    mu = np.array(scales)
    # TODO: make the variance an input parameter
    var = np.array([1000.0, 1000.0, 1.0, 1.0, 1.0])
    count = tf.Variable(1.0)  # Use tf.Variable with scalar value

    norm.set_weights([mu, var, count])  # Now provide all 3 weights

    return norm
