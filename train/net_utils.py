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


def loss_cross_entropy(target, output):
    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output)
    return tf.reduce_mean(input_tensor=policy_cross_entropy)
