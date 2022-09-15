#!/usr/bin/env python3

import json

import numpy as np
import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras

from trainable_game import TrainableGame
import net_utils

_LEARNING_RATE = 0.001


class NetType:
    SimpleTwoHeaded = "simple_two_headed"


class Hex(TrainableGame):

    BOARD_SIZE = 11
    PLANES_NUM = 2
    MOVE_NUM = BOARD_SIZE * BOARD_SIZE

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_entry = json.load(f)

        # planes of 128bit are saved in json as two 64bit values

        # convert 2 64bit planes to 128bit plane
        # data_entry["planes"] = [(a + (b << 64)) for (a, b) in zip(
        #     data_entry["planes"][::2], data_entry["planes"][1::2])]

        # we convert them into 32bits np arrays
        planes = np.frombuffer(np.array(
            data_entry["planes"], dtype=np.uint64), dtype=np.uint32).reshape((-1, 4))
        assert len(planes) == self.PLANES_NUM

        probs = np.array(data_entry["probs"], dtype=np.float32)
        assert len(probs) == self.MOVE_NUM

        winner = np.float32(data_entry["winner"])

        return (planes, probs, winner)

    def _create_model_simple_two_headed(self):
        input_layer = Input(
            shape=(self.PLANES_NUM, self.BOARD_SIZE, self.BOARD_SIZE),
            name="in_position")
        x = Dense(units=121, activation="relu")(input_layer)
        output_scalar_layer = Dense(
            units=1, activation="tanh", name="out_value")(x)
        output_probs_layer = Dense(
            units=self.MOVE_NUM, activation="sigmoid", name="out_probs")(x)

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
        if net_type == NetType.SimpleTwoHeaded:
            return self._create_model_simple_two_headed()
        else:
            raise ValueError("Unknown model type: " + net_type)

    def load_model(self, path: str, net_type: str) -> keras.Model:
        if net_type == NetType.SimpleTwoHeaded:
            custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy
            }
        else:
            raise ValueError("Unknown model type: " + net_type)

        return tf.keras.models.load_model(path, custom_objects=custom_objects)
