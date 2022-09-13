#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from hex import Hex
from tictactoe import TicTacToe
from trainable_game import NetCategory
from data_parser import DataParser


TRAIN_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TRAIN_DIR, ".."))


def main(config):
    # Organize all arguments

    working_area = config["working_area"]
    if "{RL_TOP}" in working_area:
        working_area = working_area.replace("{RL_TOP}", RL_TOP)
    working_area = Path(working_area)
    assert(working_area.exists())

    config["games_dir"] = working_area / "games"
    config["games_dir"].mkdir(exist_ok=True)
    config["models_dir"] = working_area / "models"
    config["models_dir"].mkdir(exist_ok=True)

    if config["game"] == "tictactoe":
        game = TicTacToe()
    elif config["game"] == "hex":
        game = Hex()
    else:
        raise ValueError("Unknown game argument in config file.")
    config["self_play_exec"] = config["self_play_exec"].format(config["game"])

    net_type = config["model_type"]
    base_model_path = config["base_model"]
    if base_model_path == "[none]":
        model = game.create_model(net_type)
        base_model_path = _save_model(model, config["models_dir"])
    elif base_model_path == "[latest]":
        logging.warning("Choosing latest model based on directory name format")
        all_models = list(config["models_dir"].iterdir())
        if len(all_models) == 0:
            raise ValueError(
                "Model [latest] was requested, but no existing models were found.")
        base_model_path = sorted(all_models)[-1]

    # Run training loop
    # TODO better organize the mode type parameter, and validate that the model path matches the requested model type
    play_and_train_loop(game, base_model_path, net_type, config)


def play_and_train_loop(game, base_model_path, net_type, config):
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # In each iteration there will be a new model_path
    model_path = base_model_path
    for iter_num in range(config["iterations"]):
        logging.info(f"Training iteration {iter_num}")
        training_games_dir = os.path.join(
            config["games_dir"], f"{run_id}_{iter_num}_{_model_id(game.load_model(model_path, net_type))}")

        self_play(game, model_path, training_games_dir, config)
        new_model = train(game, model_path, net_type, training_games_dir)

        new_model_path = _save_model(new_model, config["models_dir"])
        compare_models(model_path, new_model_path)
        model_path = new_model_path


def self_play(game, model_path, out_dir, config):
    profile = "dev" if config["debug"] == "true" else "release"
    subprocess.run([
        "cargo", "run", "--profile", profile, "--bin",
        config["self_play_exec"], "--",
        "--model-path", model_path,
        "--net-type", game.get_net_category(config["model_type"]) + '_net',
        "--games-num", str(config["self_play_games_num"]),
        "--out-dir", out_dir,
        "--sim-count", str(config["mcts_cfg"]["sim_count"]),
        "--explore-factor", str(config["mcts_cfg"]["explore_factor"])],
        stderr=sys.stderr, stdout=sys.stdout, check=True)


def train(game, model_path, net_type, training_games_dir):
    net_category = game.get_net_category(net_type)
    if net_category != NetCategory.TwoHeaded:
        raise ValueError("only two headed network is supported")

    logging.debug("Loading current model")
    model = game.load_model(model_path, net_type)
    xs, ys = [], []

    logging.debug("Loading games by current model")

    parser = DataParser(game, training_games_dir)

    train_dataset = tf.data.Dataset.from_generator(
        parser.generator, output_types=(tf.string, tf.string, tf.string))
    train_dataset = train_dataset.map(parser.get_parse_func())
    # train_dataset = train_dataset.batch(32, drop_remainder=True)
    train_dataset = train_dataset.prefetch(4)

    logging.info("Fitting new model...")
    model.fit(train_dataset, epochs=4, verbose=0)

    return model


def compare_models(model1_path, model2_path):
    pass  # TODO


def _save_model(model, models_dir):
    model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"model_{model_time}")
    model.save(model_path, save_format='tf')
    return model_path


def _model_id(model):
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
    for vars_ in model.trainable_variables:
        h = h * 31 + np_array_hash(vars_.numpy())
        h = h & 0xffffffffffffffff

    assert type(h) is int
    assert h <= 0xffffffffffffffff
    return h


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_ = json.load(config_file)
    main(config_)
