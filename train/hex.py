#!/usr/bin/env python3

import json

import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras

from trainable_game import TrainableGame, NetCategory
import net_utils

_LEARNING_RATE = 0.001


class NetType:
    SimpleScalar = "simple_scalar"
    SimpleTwoHeaded = "simple_two_headed"


class Hex(TrainableGame):

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_obj = json.load(f)
        return data_obj

    def _create_model_simple_scalar(self):
        input_layer = Input(shape=121, name="in_position")
        x = Dense(units="121", activation="relu")(input_layer)
        output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

        model = Model(inputs=input_layer, outputs=[output_layer])

        opt = optimizers.Adam(learning_rate=_LEARNING_RATE)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def _create_model_simple_two_headed(self):
        input_layer = Input(shape=121, name="in_position")
        x = Dense(units="121", activation="relu")(input_layer)
        output_scalar_layer = Dense(
            units="1", activation="tanh", name="out_value")(x)
        output_probs_layer = Dense(
            units="121", activation="sigmoid", name="out_probs")(x)

        model = Model(inputs=input_layer, outputs=[
            output_scalar_layer, output_probs_layer])

        opt = optimizers.Adam(learning_rate=_LEARNING_RATE)
        model.compile(
            optimizer=opt,
            loss={
                'out_value': tf.keras.losses.MeanSquaredError(),
                'out_probs': net_utils.loss_cross_entropy
            },
            metrics={'out_value': tf.keras.metrics.RootMeanSquaredError(),
                     'out_probs': tf.keras.metrics.KLDivergence()})
        return model

    def create_model(self, net_type: str) -> keras.Model:
        if net_type == NetType.SimpleScalar:
            return self._create_model_simple_scalar()
        elif net_type == NetType.SimpleTwoHeaded:
            return self._create_model_simple_two_headed()
        else:
            raise ValueError("Unknown model type: " + net_type)

    def load_model(self, path: str, net_type: str) -> keras.Model:
        if net_type == NetType.SimpleScalar:
            custom_objects = {}
        elif net_type == NetType.SimpleTwoHeaded:
            custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy
            }
        else:
            raise ValueError("Unknown model type: " + net_type)

        return tf.keras.models.load_model(path, custom_objects=custom_objects)

    def get_net_category(self, net_type: str) -> NetCategory:
        if net_type == NetType.SimpleScalar:
            return NetCategory.Scalar
        elif net_type == NetType.SimpleTwoHeaded:
            return NetCategory.TwoHeaded
        else:
            raise ValueError("Unknown model type: " + net_type)
