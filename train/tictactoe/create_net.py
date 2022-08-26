#!/usr/bin/env python3

import tensorflow as tf
from keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

LEARNING_RATE = 0.001


def create_model_simple_scalar():
    input_layer = Input(shape=9, name="in_position")
    x = Dense(units="9", activation="relu")(input_layer)
    output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

    model = Model(inputs=input_layer, outputs=[output_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def create_model_simple_two_headed():
    input_layer = Input(shape=9, name="in_position")
    x = Dense(units="9", activation="relu")(input_layer)
    output_scalar_layer = Dense(
        units="1", activation="tanh", name="out_value")(x)
    output_probs_layer = Dense(
        units="9", activation="sigmoid", name="out_probs")(x)

    model = Model(inputs=input_layer, outputs=[
                  output_scalar_layer, output_probs_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=opt,
        loss={'out_value': 'mse', 'out_probs': 'kl_divergence'},
        metrics={'out_value': tf.keras.metrics.RootMeanSquaredError(),
                 'out_probs': tf.keras.metrics.KLDivergence()})
    return model

