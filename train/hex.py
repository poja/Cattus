import json

import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model

from trainable_game import TrainableGame

_LEARNING_RATE = 0.001


class Hex(TrainableGame):

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_obj = json.load(f)

        # Network always accept position as
        if data_obj["turn"] != 1:
            data_obj = self._flip_position(data_obj)
        assert data_obj["turn"] == 1

        return data_obj

    def create_model_simple_scalar(self):
        input_layer = Input(shape=121, name="in_position")
        x = Dense(units="121", activation="relu")(input_layer)
        output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

        model = Model(inputs=input_layer, outputs=[output_layer])

        opt = optimizers.Adam(learning_rate=_LEARNING_RATE)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def create_model_simple_two_headed(self):
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
            loss={'out_value': 'mse', 'out_probs': 'kl_divergence'},
            metrics={'out_value': tf.keras.metrics.RootMeanSquaredError(),
                     'out_probs': tf.keras.metrics.KLDivergence()})
        return model

    @staticmethod
    def _flip_position(data_obj):
        flipped_data_obj = {}
        flipped_data_obj["turn"] = -data_obj["turn"]
        flipped_data_obj["winner"] = -data_obj["winner"]

        flipped_data_obj["position"] = [None] * 121
        for idx, val in enumerate(data_obj["position"]):
            r, c = idx // 11, idx % 11
            flipped_idx = c * 11 + r
            flipped_data_obj["position"][flipped_idx] = -val

        flipped_data_obj["moves_probabilities"] = [None] * 121
        for idx, val in enumerate(data_obj["moves_probabilities"]):
            r, c = idx // 11, idx % 11
            flipped_idx = c * 11 + r
            flipped_data_obj["moves_probabilities"][flipped_idx] = val

        return flipped_data_obj
