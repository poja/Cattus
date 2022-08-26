#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import struct
import subprocess
import sys

import numpy as np
import tensorflow as tf

from tictactoe import ttt_utils
from tictactoe.create_net import create_model_simple_two_headed
from trainable_game import NetType

BATCH_SIZE = 4
EPOCHS = 16
LEARNING_RATE = 0.001


def main(config):
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    base_model_path = config["base_model"]
    if base_model_path is None:
        model = create_model_simple_two_headed()
        base_model_path = _save_model(model, config['models_dir'])

    model_path = base_model_path

    for iter_num in range(config["iterations"]):
        logging.info(f"Training iteration {iter_num}")
        data_dir = os.path.join(
            config["data_dir"], f"{run_id}_{iter_num}_{_model_id(model_path)}")

        self_play(model_path, data_dir, config)
        new_model = train(model_path, data_dir)

        new_model_path = _save_model(new_model, config['models_dir'])
        compare_models(model_path, new_model_path)
        model_path = new_model_path


def self_play(model_path, out_dir, config):
    subprocess.run([config["self_play_exec"],
                    "--model-path", model_path,
                    "--net-type", "two_headed_net",
                    "--games-num", str(config["self_play_games_num"]),
                    "--out-dir", out_dir,
                    "--sim-count", str(config["mcts_cfg"]["sim_count"]),
                    "--explore-factor", str(config["mcts_cfg"]["explore_factor"])],
                   stderr=sys.stderr, stdout=sys.stdout, check=True)


def train(model_path, data_dir):
    logging.debug('Loading current model')

    model = tf.keras.models.load_model(model_path)
    xs, ys = [], []

    logging.debug('Loading games by current model')

    nettype = NetType.SimpleTwoHeaded
    for data_file in os.listdir(data_dir):
        data_filename = os.path.join(data_dir, data_file)
        data_entry = ttt_utils.load_data_entry(data_filename)
        xs.append(data_entry["position"])
        if nettype == NetType.SimpleScalar:
            ys.append(data_entry["winner"])
        elif nettype == NetType.SimpleTwoHeaded:
            y = (data_entry["winner"], data_entry["moves_probabilities"])
            ys.append(y)
        else:
            raise ValueError("Unknown model type: " + nettype)

    xs = np.array(xs)
    if nettype == NetType.SimpleScalar:
        ys = np.array(ys)
    elif nettype == NetType.SimpleTwoHeaded:
        ys_raw = ys
        ys = {'out_value': np.array([y[0] for y in ys_raw]),
              'out_probs': np.array([y[1] for y in ys_raw])}

    logging.info('Fitting new model...')
    model.fit(x=xs, y=ys, batch_size=BATCH_SIZE, epochs=EPOCHS)

    return model


def compare_models(model1_path, model2_path):
    pass  # TODO


def _save_model(model, models_dir):
    model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f'model_{model_time}')
    model.save(model_path, save_format='tf')
    return model_path


def _load_model(path):
    return tf.keras.models.load_model(path)


def _model_id(model_path):
    def floatToBits(f):
        return struct.unpack('>l', struct.pack('>f', f))[0]

    def np_array_hash(arr):
        h = 0
        for a in arr:
            h = h * 31 + (np_array_hash(a) if type(a)
                                              is np.ndarray else floatToBits(a))
            h = h & 0xffffffffffffffff
        return h

    h = 0
    for vars in _load_model(model_path).trainable_variables:
        h = h * 31 + np_array_hash(vars.numpy())
        h = h & 0xffffffffffffffff

    assert type(h) is int
    assert h <= 0xffffffffffffffff
    return h


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_ = json.load(config_file)
    main(config_)
