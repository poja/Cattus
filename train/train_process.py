#!/usr/bin/env python3


import datetime
import logging
import os
import subprocess
import sys
import random
from pathlib import Path
import copy
import json
import tensorflow as tf
import tempfile
import shutil

from hex import Hex
from tictactoe import TicTacToe
from chess import Chess
from data_parser import DataParser
import net_utils


TRAIN_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TRAIN_DIR, ".."))


class TrainProcess:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)

        working_area = self.cfg["working_area"]
        working_area = working_area.format(RL_TOP=RL_TOP)
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
        elif self.cfg["game"] == "chess":
            self.game = Chess()
        else:
            raise ValueError("Unknown game argument in config file.")
        self.self_play_exec = "{}_self_play_runner".format(self.cfg["game"])

        self.net_type = self.cfg["model"]["type"]
        base_model_path = self.cfg["model"]["base"]
        if base_model_path == "[none]":
            model = self.game.create_model(self.net_type, self.cfg)
            assert model.get_layer("value_head").output_shape == (None, 1)
            assert model.get_layer("policy_head").output_shape == (
                None, self.game.MOVE_NUM)
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
        games_dir = os.path.join(self.cfg["games_dir"], run_id)

        best_model = self.base_model_path
        latest_model = self.base_model_path
        for iter_num in range(self.cfg["self_play"]["iterations"]):
            logging.info(f"Training iteration {iter_num}")

            self._self_play(best_model, games_dir, iter_num)
            new_model = self._train(latest_model, games_dir)

            latest_model = self._save_model(new_model)
            w1, w2, d = self._compare_models(best_model, latest_model)
            total_games = w1 + w2 + d
            winning = w2 / total_games
            losing = w1 / total_games
            if winning > self.cfg["training"]["compare"]["switching_winning_threshold"]:
                best_model = latest_model
            elif losing > self.cfg["training"]["compare"]["warning_losing_threshold"]:
                print(
                    "WARNING: new model is worse than previous one, losing ratio:", losing)
            print("best model:", best_model)

    def _self_play(self, model_path, out_dir, iter_num):
        profile = "dev" if self.cfg["debug"] == "true" else "release"
        subprocess.run([
            "cargo", "run", "--profile", profile, "--bin",
            self.self_play_exec, "--",
            "--model1-path", model_path,
            "--model2-path", model_path,
            "--games-num", str(self.cfg["self_play"]["games_num"]),
            "--out-dir1", out_dir,
            "--out-dir2", out_dir,
            "--data-entries-prefix", "i{0:04d}_".format(iter_num),
            "--sim-count", str(self.cfg["mcts"]["sim_count"]),
            "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
            "--threads", str(self.cfg["self_play"]["threads"])],
            stderr=sys.stderr, stdout=sys.stdout, check=True)

    def _train(self, model_path, training_games_dir):
        logging.debug("Loading current model")
        model = self.game.load_model(model_path, self.net_type)

        logging.debug("Loading games by current model")
        parser = DataParser(self.game, training_games_dir, self.cfg)
        train_dataset = tf.data.Dataset.from_generator(
            parser.generator, output_types=(tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(parser.get_parse_func())
        # train_dataset = train_dataset.batch(32, drop_remainder=True)
        train_dataset = train_dataset.prefetch(4)

        logging.info("Fitting new model...")
        model.fit(train_dataset, epochs=1, verbose=0)

        return model

    def _compare_models(self, model1_path, model2_path):
        tmp_dirpath = tempfile.mkdtemp()
        try:
            compare_res_file = os.path.join(tmp_dirpath, "compare_result.json")
            games_dir = os.path.join(tmp_dirpath, "games")

            profile = "dev" if self.cfg["debug"] == "true" else "release"
            subprocess.run([
                "cargo", "run", "--profile", profile, "--bin",
                self.self_play_exec, "--",
                "--model1-path", model1_path,
                "--model2-path", model2_path,
                "--games-num", str(self.cfg["training"]
                                   ["compare"]["games_num"]),
                "--out-dir1", games_dir,
                "--out-dir2", games_dir,
                "--result-file", compare_res_file,
                "--sim-count", str(self.cfg["mcts"]["sim_count"]),
                "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
                "--threads", str(self.cfg["training"]["compare"]["threads"])],
                stderr=sys.stderr, stdout=sys.stdout, check=True)

            with open(compare_res_file, "r") as res_file:
                res = json.load(res_file)
            return res["player1_wins"], res["player2_wins"], res["draws"]
        finally:
            shutil.rmtree(tmp_dirpath)

    def _save_model(self, model):
        model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + \
            "_{0:04x}".format(random.randint(0, 1 << 16))
        model_path = os.path.join(
            self.cfg["models_dir"], f"model_{model_time}")
        model.save(model_path, save_format='tf')
        return model_path
