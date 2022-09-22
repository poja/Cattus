#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import struct


def model_id(model):
    def floatToBits(f):
        return struct.unpack('>l', struct.pack('>f', f))[0]

    def np_array_hash(arr):
        h = 0
        for a in arr:
            h = h * 31 + (np_array_hash(a) if type(a)
                          is np.ndarray else floatToBits(a))
            h = h & 0xffffffffffffffff
        return h

    h = 0
    for vars_ in model.trainable_variables:
        h = h * 31 + np_array_hash(vars_.numpy())
        h = h & 0xffffffffffffffff

    assert type(h) is int
    assert h <= 0xffffffffffffffff
    return h


# The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW
# N: number of images in the batch
# H: height of the image
# W: width of the image
# C: number of channels of the image
#
# When running on CPU need to change to 'channels_last'
#


def loss_cross_entropy(target, output):
    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output)
    return tf.reduce_mean(input_tensor=policy_cross_entropy)


def batch_norm(input, name, scale=False):
    return tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        axis=1,  # ?
        center=True,
        scale=scale,
        name=name)(input)


def conv_block(inputs,
               filter_size,
               output_channels,
               name,
               l2reg,
               cpu,
               bn_scale=False):
    conv_data_fmt = 'channels_last' if cpu else 'channels_first'

    # convolution
    flow = tf.keras.layers.Conv2D(output_channels,
                                  filter_size,
                                  use_bias=False,
                                  padding='same',
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=l2reg,
                                  data_format=conv_data_fmt,
                                  name=name + '/conv2d')(inputs)
    # batch normalization
    flow = batch_norm(flow, name=name + '/bn', scale=bn_scale)
    # a rectifier nonlinearity
    return tf.keras.layers.Activation('relu')(flow)


def residual_block(inputs, channels, name, l2reg, cpu):
    conv_data_fmt = 'channels_last' if cpu else 'channels_first'

    # convolution
    flow = tf.keras.layers.Conv2D(channels,
                                  3,
                                  use_bias=False,
                                  padding='same',
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=l2reg,
                                  data_format=conv_data_fmt,
                                  name=name + '/1/conv2d')(inputs)
    # batch normalization
    flow = batch_norm(flow, name + '/1/bn', scale=False)
    # a rectifier nonlinearity
    flow = tf.keras.layers.Activation('relu')(flow)

    # convolution
    flow = tf.keras.layers.Conv2D(channels,
                                  3,
                                  use_bias=False,
                                  padding='same',
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=l2reg,
                                  data_format=conv_data_fmt,
                                  name=name + '/2/conv2d')(flow)
    # batch normalization
    flow = batch_norm(flow, name + '/2/bn', scale=True)
    # ... (squeeze_excitation)
    #  skip connection adding input to the block
    flow = tf.keras.layers.add([inputs, flow])

    # a rectifier nonlinearity
    return tf.keras.layers.Activation('relu')(flow)


def create_convnetv1(inputs, residual_filter_num, residual_block_num, moves_num, l2reg, cpu):
    l2reg = tf.keras.regularizers.l2(l=l2reg)

    # single conv block
    flow = conv_block(inputs,
                      filter_size=3,
                      output_channels=residual_filter_num,
                      name='in_position',
                      l2reg=l2reg,
                      cpu=cpu,
                      bn_scale=True)

    # multiple residual blocks
    for block_idx in range(residual_block_num):
        flow = residual_block(flow,
                              residual_filter_num,
                              name='residual_{}'.format(block_idx + 1),
                              l2reg=l2reg,
                              cpu=cpu)

    # Value head
    flow_val = conv_block(flow,
                          filter_size=1,
                          output_channels=2,
                          name='value',
                          l2reg=l2reg,
                          cpu=cpu)
    flow_val = tf.keras.layers.Flatten()(flow_val)
    flow_val = tf.keras.layers.Dense(128,
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=l2reg,
                                     activation='relu',
                                     name='value/dense1')(flow_val)
    head_val = tf.keras.layers.Dense(1,
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=l2reg,
                                     activation='tanh',
                                     name='out_value')(flow_val)

    # Policy head
    flow_pol = conv_block(flow,
                          filter_size=1,
                          output_channels=2,  # accept as argument
                          name='policy',
                          l2reg=l2reg,
                          cpu=cpu)
    flow_pol = tf.keras.layers.Flatten()(flow_pol)
    head_pol = tf.keras.layers.Dense(moves_num,
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=l2reg,
                                     bias_regularizer=l2reg,
                                     name='out_probs')(flow_pol)

    return [head_val, head_pol]
