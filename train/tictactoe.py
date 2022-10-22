#!/usr/bin/env python3

import json

import numpy as np
import tensorflow as tf
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
import keras

from train.trainable_game import TrainableGame
from train import net_utils


class TtoNetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class TicTacToe(TrainableGame):
    BOARD_SIZE = 3
    PLANES_NUM = 3
    MOVE_NUM = BOARD_SIZE * BOARD_SIZE

    def load_data_entry(self, path):
        with open(path, "rb") as f:
            data_entry = json.load(f)

        planes = np.array(data_entry["planes"],
                          dtype=np.uint32).reshape((self.PLANES_NUM))
        assert len(planes) == self.PLANES_NUM

        probs = np.array(data_entry["probs"], dtype=np.float32)
        assert len(probs) == self.MOVE_NUM

        winner = np.float32(data_entry["winner"])

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
        flow = Dense(units=9, activation="relu")(flow)
        head_val = Dense(
            units=1, activation="tanh", name="value_head")(flow)
        head_probs = Dense(units=self.MOVE_NUM, name="policy_head")(flow)
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
            residual_filter_num=cfg["model"]["residual_filter_num"],
            residual_block_num=cfg["model"]["residual_block_num"],
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
        if net_type == TtoNetType.SimpleTwoHeaded:
            return self._create_model_simple_two_headed(cfg)
        elif net_type == TtoNetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def load_model(self, path: str, net_type: str) -> keras.Model:
        if net_type == TtoNetType.SimpleTwoHeaded or net_type == TtoNetType.ConvNetV1:
            custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy,
                "value_head_accuracy": net_utils.value_head_accuracy}
        else:
            raise ValueError("Unknown model type: " + net_type)

        return tf.keras.models.load_model(path, custom_objects=custom_objects)
