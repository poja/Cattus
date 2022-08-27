import json

import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model

from trainable_game import TrainableGame

_LEARNING_RATE = 0.001


class TicTacToe(TrainableGame):

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_obj = json.load(f)

        # Network always accept position as
        if data_obj["turn"] != 1:
            data_obj = self._flip_position(data_obj)
        assert data_obj["turn"] == 1

        return data_obj

    def create_model_simple_scalar(self):
        input_layer = Input(shape=9, name="in_position")
        x = Dense(units="9", activation="relu")(input_layer)
        output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

        model = Model(inputs=input_layer, outputs=[output_layer])

        opt = optimizers.Adam(learning_rate=_LEARNING_RATE)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def create_model_simple_two_headed(self):
        input_layer = Input(shape=9, name="in_position")
        x = Dense(units="9", activation="relu")(input_layer)
        output_scalar_layer = Dense(
            units="1", activation="tanh", name="out_value")(x)
        output_probs_layer = Dense(
            units="9", activation="sigmoid", name="out_probs")(x)

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
        return {
            "turn": -data_obj["turn"],
            "winner": -data_obj["winner"],
            "position": [-x for x in data_obj["position"]],
            "moves_probabilities": data_obj["moves_probabilities"]
        }
