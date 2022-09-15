#!/usr/bin/env python3


import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path
import copy
import json
import tensorflow as tf

from hex import Hex
from tictactoe import TicTacToe
from data_parser import DataParser
import net_utils


TRAIN_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TRAIN_DIR, ".."))


class TrainProcess:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)

        working_area = self.cfg["working_area"]
        if "{RL_TOP}" in working_area:
            working_area = working_area.replace("{RL_TOP}", RL_TOP)
        self.cfg["working_area"] = working_area
        working_area = Path(working_area)
        assert(working_area.exists())

        self.cfg["games_dir"] = working_area / "games"
        self.cfg["games_dir"].mkdir(exist_ok=True)
        self.cfg["models_dir"] = working_area / "models"
        self.cfg["models_dir"].mkdir(exist_ok=True)

        if self.cfg["game"] == "tictactoe":
            self.game = TicTacToe()
        elif self.cfg["game"] == "hex":
            self.game = Hex()
        else:
            raise ValueError("Unknown game argument in config file.")
        self.cfg["self_play"]["exec"] = self.cfg["self_play"]["exec"].format(
            self.cfg["game"])
        self.cfg["training"]["compare"]["exec"] = self.cfg["training"]["compare"]["exec"].format(
            self.cfg["game"])

        self.net_type = self.cfg["model"]["type"]
        base_model_path = self.cfg["model"]["base"]
        if base_model_path == "[none]":
            model = self.game.create_model(self.net_type, self.cfg["model"])
            base_model_path = self._save_model(model)
        elif base_model_path == "[latest]":
            logging.warning(
                "Choosing latest model based on directory name format")
            all_models = list(self.cfg["models_dir"].iterdir())
            if len(all_models) == 0:
                raise ValueError(
                    "Model [latest] was requested, but no existing models were found.")
            base_model_path = sorted(all_models)[-1]
        self.base_model_path = base_model_path

    def run_training_loop(self):
        # TODO better organize the mode type parameter, and validate that the model path matches the requested model type
        run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

        # In each iteration there will be a new model_path
        model_path = self.base_model_path
        for iter_num in range(self.cfg["self_play"]["iterations"]):
            logging.info(f"Training iteration {iter_num}")
            model_id = net_utils.model_id(
                self.game.load_model(model_path, self.net_type))
            training_games_dir = os.path.join(
                self.cfg["games_dir"], f"{run_id}_{iter_num}_{model_id}")

            self._self_play(model_path, training_games_dir)
            new_model = self._train(model_path, training_games_dir)

            new_model_path = self._save_model(new_model)
            w1, w2, d = self._compare_models(model_path, new_model_path)
            total_games = w1 + w2 + d
            winning = w2 / total_games
            losing = w1 / total_games
            if winning > self.cfg["training"]["compare"]["switching_winning_threshold"]:
                model_path = new_model_path
            elif losing > self.cfg["training"]["compare"]["warning_losing_threshold"]:
                print("WARNING: new model is worse than previous one, losing ratio:", losing)

    def _self_play(self, model_path, out_dir):
        profile = "dev" if self.cfg["debug"] == "true" else "release"
        subprocess.run([
            "cargo", "run", "--profile", profile, "--bin",
            self.cfg["self_play"]["exec"], "--",
            "--model-path", model_path,
            "--games-num", str(self.cfg["self_play"]["games_num"]),
            "--out-dir", out_dir,
            "--sim-count", str(self.cfg["mcts"]["sim_count"]),
            "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
            "--threads", str(self.cfg["self_play"]["threads"])],
            stderr=sys.stderr, stdout=sys.stdout, check=True)

    def _train(self, model_path, training_games_dir):
        logging.debug("Loading current model")
        model = self.game.load_model(model_path, self.net_type)

        logging.debug("Loading games by current model")
        parser = DataParser(self.game, training_games_dir,
                            self.cfg["training"]["entries_count"])
        train_dataset = tf.data.Dataset.from_generator(
            parser.generator, output_types=(tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(parser.get_parse_func())
        # train_dataset = train_dataset.batch(32, drop_remainder=True)
        train_dataset = train_dataset.prefetch(4)

        logging.info("Fitting new model...")
        model.fit(train_dataset, epochs=4, verbose=0)

        return model

    def _compare_models(self, model1_path, model2_path):
        compare_res_file = os.path.join(self.cfg["working_area"], "compare_result.json")

        profile = "dev" if self.cfg["debug"] == "true" else "release"
        subprocess.run([
            "cargo", "run", "--profile", profile, "--bin",
            self.cfg["training"]["compare"]["exec"], "--",
            "--model1-path", model1_path,
            "--model2-path", model2_path,
            "--games-num", str(self.cfg["training"]["compare"]["games_num"]),
            "--result-file", compare_res_file,
            "--sim-count", str(self.cfg["mcts"]["sim_count"]),
            "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
            "--threads", str(self.cfg["training"]["compare"]["threads"])],
            stderr=sys.stderr, stdout=sys.stdout, check=True)

        with open(compare_res_file, "r") as res_file:
            res = json.load(res_file)
        return res["player1_wins"], res["player2_wins"], res["draws"]

    def _save_model(self, model):
        model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        model_path = os.path.join(
            self.cfg["models_dir"], f"model_{model_time}")
        model.save(model_path, save_format='tf')
        return model_path
