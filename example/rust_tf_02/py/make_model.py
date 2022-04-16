#!/bin/usr/env python3

import argparse
import tensorflow as tf


def make_model(output_dir):
    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    inputs = tf.placeholder(tf.float32, shape=[2], name="inputs")
    labels = tf.placeholder(tf.int32, shape=[], name="labels")

    hidden = tf.layers.Dense(20, activation=tf.nn.relu)(inputs)
    logits = tf.squeeze(tf.layres.Dense(1)(hidden))

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits, name="loss")
    optimizer = tf.train.GradientDescentOptimizer(
        lr).minimize(loss, name="train")

    predications = tf.cast(tf.round(tf.nn.sigmoid(logits)),
                           tf.int32, name="predicated")
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(labels, predications), tf.float32), name="accuracy")

    # with tf.Graph().as_default(), tf.Session() as session:
    #     with tf.variable_scope("model"):
    #         m = create


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model creator")
    parser.add_argument("--out", dest="output_dir", type=str,
                        required=True, help="output directory")
    args = parser.parse_args()

    make_model(args.output_dir)
