#!/usr/bin/env python3

import tensorflow as tf


def loss_cross_entropy(target, output):
    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output)
    return tf.reduce_mean(input_tensor=policy_cross_entropy)
