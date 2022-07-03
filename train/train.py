#!/usr/bin/env python3

import os
import tensorflow as tf
import datetime
import json
import time
import argparse
import subprocess
import numpy as np
import struct
import sys


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


def train(model_path, data_dir, config):
    pass  # TODO


def main(config):
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    best_model_path = config["base_model"]

    for iter_num in range(config["iterations"]):
        print(f"Iteration {iter_num}")  # TODO remove

        model_path = best_model_path
        data_dir = os.path.join(
            config["data_dir"], f"{run_id}_{iter_num}_{model_id(model_path)}")
        self_play(model_path, data_dir, config)
        train(model_path, data_dir, config)

        # TODO compare trained model with best model and switch if better


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    main(config)
