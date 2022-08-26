#!/usr/bin/env python3

import tensorflow as tf
from keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

LEARNING_RATE = 0.001


class NetType:
    SimpleScalar = "simple_scalar"
    SimpleTwoHeaded = "simple_two_headed"


def create_model_simple_scalar():
    input_layer = Input(shape=121, name="in_position")
    x = Dense(units="121", activation="relu")(input_layer)
    output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

    model = Model(inputs=input_layer, outputs=[output_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def create_model_simple_two_headed():
    input_layer = Input(shape=121, name="in_position")
    x = Dense(units="121", activation="relu")(input_layer)
    output_scalar_layer = Dense(
        units="1", activation="tanh", name="out_value")(x)
    output_probs_layer = Dense(
        units="121", activation="sigmoid", name="out_probs")(x)

    model = Model(inputs=input_layer, outputs=[
                  output_scalar_layer, output_probs_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=opt,
        loss={'out_value': 'mse', 'out_probs': 'kl_divergence'},
        metrics={'out_value': tf.keras.metrics.RootMeanSquaredError(),
                 'out_probs': tf.keras.metrics.KLDivergence()})
    return model

