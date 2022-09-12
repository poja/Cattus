import json

import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras

from trainable_game import TrainableGame, NetCategory

_LEARNING_RATE = 0.001


class NetType:
    SimpleScalar = "simple_scalar"
    SimpleTwoHeaded = "simple_two_headed"


class TicTacToe(TrainableGame):

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_obj = json.load(f)

        # Network always accept position as
        if data_obj["turn"] != 1:
            data_obj = self._flip_position(data_obj)
        assert data_obj["turn"] == 1

        return data_obj

    def _create_model_simple_scalar(self):
        input_layer = Input(shape=9, name="in_position")
        x = Dense(units="9", activation="relu")(input_layer)
        output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

        model = Model(inputs=input_layer, outputs=[output_layer])

        opt = optimizers.Adam(learning_rate=_LEARNING_RATE)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def _create_model_simple_two_headed(self):
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
            metrics={'out_value': tf.keras.metrics.MeanSquaredError(),
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
        return tf.keras.models.load_model(path)

    def get_net_category(self, net_type: str) -> NetCategory:
        if net_type == NetType.SimpleScalar:
            return NetCategory.Scalar
        elif net_type == NetType.SimpleTwoHeaded:
            return NetCategory.TwoHeaded
        else:
            raise ValueError("Unknown model type: " + net_type)

    @staticmethod
    def _flip_position(data_obj):
        return {
            "turn": -data_obj["turn"],
            "winner": -data_obj["winner"],
            "position": [-x for x in data_obj["position"]],
            "moves_probabilities": data_obj["moves_probabilities"]
        }
