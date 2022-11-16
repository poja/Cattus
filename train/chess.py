import sys
import numpy as np
import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras
from construct import Struct, Array, Int64ul, Float32l, Int8sl, Int8ul

from train.trainable_game import TrainableGame
from train import net_utils


class NetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class Chess(TrainableGame):
    BOARD_SIZE = 8
    PLANES_NUM = 18
    MOVE_NUM = 1880

    ENTRY_FORMAT = Struct(
        "planes" / Array(PLANES_NUM, Int64ul),
        "moves_bitmap" / Array(235, Int8ul),
        "probs" / Array(225, Float32l),
        "winner" / Int8sl,
    )

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            entry_bytes = f.read()
        assert len(entry_bytes) == self.ENTRY_FORMAT.sizeof(), "invalid training data file: {} ({} != {})".format(
            path, len(entry_bytes), self.ENTRY_FORMAT.sizeof())
        entry = self.ENTRY_FORMAT.parse(entry_bytes)
        planes = np.array(entry.planes, dtype=np.uint64)
        moves_bitmap = np.array(entry.moves_bitmap, dtype=np.uint8)
        probs = np.array(entry.probs, dtype=np.float32)
        winner = entry.winner

        probs_all = np.full((self.MOVE_NUM), -1.0, dtype=np.float32)
        probs_idx = 0
        for move_idx in range(self.MOVE_NUM):
            i, j = move_idx // 8, move_idx % 8
            if moves_bitmap[i] & (1 << j) != 0:
                probs_all[move_idx] = probs[probs_idx]
                probs_idx += 1
        probs = probs_all

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
        x = Dense(units=128, activation="relu")(flow)
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
            value_head_conv_output_channels_num=cfg["model"]["value_head_conv_output_channels_num"],
            policy_head_conv_output_channels_num=cfg["model"]["policy_head_conv_output_channels_num"],
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
        if net_type == NetType.SimpleTwoHeaded:
            return self._create_model_simple_two_headed(cfg)
        elif net_type == NetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def load_model(self, path: str, net_type: str) -> keras.Model:
        if net_type == NetType.SimpleTwoHeaded or net_type == NetType.ConvNetV1:
            custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy,
                "value_head_accuracy": net_utils.value_head_accuracy}
        else:
            raise ValueError("Unknown model type: " + net_type)

        return tf.keras.models.load_model(path, custom_objects=custom_objects)
