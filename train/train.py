#!/usr/bin/env python3

import os
from pathlib import Path
import tensorflow as tf
import datetime
import json
import argparse
import subprocess
import numpy as np
import struct
import sys
import random
import logging


def load_model(path):
    return tf.keras.models.load_model(path)


def model_id(model_path):
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
    for vars in load_model(model_path).trainable_variables:
        h = h * 31 + np_array_hash(vars.numpy())
        h = h & 0xffffffffffffffff

    assert type(h) is int
    assert h <= 0xffffffffffffffff
    return h


def self_play(model_path, out_dir, config):
    subprocess.run([config["self_play_exec"],
                    "--model", model_path,
                    "--games-num", str(config["self_play_games_num"]),
                    "--out-dir",  out_dir,
                    "--sim-count", str(config["mcts_cfg"]["sim_count"]),
                    "--explore-factor", str(config["mcts_cfg"]["explore_factor"])],
                   stderr=sys.stderr, stdout=sys.stdout, check=True)


def train(model_path, config):

    def reverse_position(pos, winner):
        return [-x for x in pos], -winner

    logging.debug('Loading all games made by the current model')

    data_dir = Path(config["data_dir"])
    positions = []
    winners = []
    for training_iteration_dir in data_dir.iterdir():
        if str(model_id(model_path)) not in str(training_iteration_dir):
            continue
        for data_file in training_iteration_dir.iterdir():
            with open(os.path.join(data_dir, data_file), "rb") as f:
                data_obj = json.load(f)
            pos, win = data_obj["position"], data_obj["winner"]
            if random.choice([True, False]):
                pos, win = reverse_position(pos, win)
            positions.append(pos)
            winners.append(win)

    logging.debug('Fitting model to newly generated games')
        
    model = load_model(model_path)
    model.fit(positions, winners, batch_size=4, epochs=16)

    logging.debug('Crude sanity check of model fitness')
    preds = [x[0] for x in model.predict(positions)]
    preds = [1 if x >= 0 else -1 for x in preds]
    wins = [1 if x >= 0 else -1 for x in winners]
    acc = [preds[i] == wins[i] for i in range(len(preds))]
    logging.info(f'Model trained. Model accuracy for training set: {sum(acc) / len(acc)}')
    
    return model



def main(config):
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    best_model_path = config["base_model"]
    model_path = best_model_path

    for iter_num in range(config["iterations"]):
        logging.info(f"Training iteration {iter_num}")
        data_dir = os.path.join(
            config["data_dir"], f"{run_id}_{iter_num}_{model_id(model_path)}")
        self_play(model_path, data_dir, config)
        new_model = train(model_path, config)
        model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        model_path = os.path.join(config["models_dir"], f'model_{model_time}')
        new_model.save(model_path, save_format='tf')

        # TODO compare trained model with best model and switch if better


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
        config = json.load(config_file)
    main(config)
