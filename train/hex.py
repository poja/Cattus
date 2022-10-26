import sys
import numpy as np
import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras
from construct import Struct, Array, Int64ul, Float32l, Int8sl

from train.trainable_game import TrainableGame
from train import net_utils


class HexNetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class Hex(TrainableGame):
    BOARD_SIZE = 11
    PLANES_NUM = 3
    MOVE_NUM = BOARD_SIZE * BOARD_SIZE

    ENTRY_FORMAT = Struct(
        "planes" / Array(PLANES_NUM * 2, Int64ul),
        "probs" / Array(MOVE_NUM, Float32l),
        "winner" / Int8sl,
    )

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            entry_bytes = f.read()
        assert self.ENTRY_FORMAT.sizeof() == len(entry_bytes)
        entry = self.ENTRY_FORMAT.parse(entry_bytes)
        # planes of 128bit are saved as two 64bit values
        planes = np.array(entry.planes, dtype=np.uint64).reshape((self.PLANES_NUM, 2))
        probs = np.array(entry.probs, dtype=np.float32)
        winner = entry.winner

        assert len(planes) == self.PLANES_NUM
        assert len(probs) == self.MOVE_NUM
        return (planes, probs, winner)

    def _get_input_shape(self, cfg):
        shape_cpu = (self.BOARD_SIZE, self.BOARD_SIZE, self.PLANES_NUM)
        shape_gpu = (self.PLANES_NUM, self.BOARD_SIZE, self.BOARD_SIZE)
        return shape_cpu if cfg["cpu"] else shape_gpu

    def _create_model_simple_two_headed(self, cfg):
        inputs = Input(
            shape=self._get_input_shape(cfg),
            name="input_planes")
        flow = tf.keras.layers.Flatten()(inputs)
        x = Dense(units=121, activation="relu")(flow)
        head_val = Dense(
            units=1, activation="tanh", name="value_head")(x)
        head_probs = Dense(units=self.MOVE_NUM, name="policy_head")(x)
        model = Model(inputs=inputs, outputs=[head_val, head_probs])

        # lr doesn't matter, will be set by train process
        opt = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=opt,
            loss={'value_head': tf.keras.losses.MeanSquaredError(),
                  'policy_head': net_utils.loss_cross_entropy},
            metrics={'value_head': net_utils.value_head_accuracy,
                     'policy_head': net_utils.policy_head_accuracy})
        return model

    def _create_model_convnetv1(self, cfg):
        inputs = Input(
            shape=self._get_input_shape(cfg),
            name="input_planes")
        outputs = net_utils.create_convnetv1(
            inputs,
            residual_block_num=cfg["model"]["residual_block_num"],
            residual_filter_num=cfg["model"]["residual_filter_num"],
            moves_num=self.MOVE_NUM,
            l2reg=cfg["model"]["l2reg"],
            cpu=cfg["cpu"])
        model = Model(inputs=inputs, outputs=outputs)

        # lr doesn't matter, will be set by train process
        opt = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=opt,
            loss={'value_head': tf.keras.losses.MeanSquaredError(),
                  'policy_head': net_utils.loss_cross_entropy},
            metrics={'value_head': net_utils.value_head_accuracy,
                     'policy_head': net_utils.policy_head_accuracy})
        return model

    def create_model(self, net_type: str, cfg) -> keras.Model:
        if net_type == HexNetType.SimpleTwoHeaded:
            return self._create_model_simple_two_headed(cfg)
        elif net_type == HexNetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def load_model(self, path: str, net_type: str) -> keras.Model:
        if net_type == HexNetType.SimpleTwoHeaded or net_type == HexNetType.ConvNetV1:
            custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy,
                "value_head_accuracy": net_utils.value_head_accuracy}
        else:
            raise ValueError("Unknown model type: " + net_type)

        return tf.keras.models.load_model(path, custom_objects=custom_objects)
