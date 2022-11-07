import time
import datetime
import logging
import os
import re
import subprocess
import sys
import random
from pathlib import Path
import copy
import json
import tempfile
import shutil
import tensorflow as tf
import keras


from train.hex import Hex
from train.tictactoe import TicTacToe
from train.chess import Chess
from train.data_parser import DataParser
from train import net_utils


TRAIN_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TRAIN_DIR, ".."))


def dictionary_to_str(d, indent=0):
    s = ""
    for key, value in d.items():
        if isinstance(value, dict):
            s += "  " * indent + str(key) + ":\n"
            s += dictionary_to_str(value, indent+1)
        else:
            s += "  " * indent + str(key) + ": \t" + str(value) + "\n"
    return s


class TrainProcess:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)

        working_area = self.cfg["working_area"]
        working_area = working_area.format(CATTUS_TOP=CATTUS_TOP)
        self.cfg["working_area"] = working_area
        working_area = Path(working_area)
        assert working_area.exists()

        self.cfg["games_dir"] = working_area / "games"
        self.cfg["games_dir"].mkdir(exist_ok=True)
        self.cfg["models_dir"] = working_area / "models"
        self.cfg["models_dir"].mkdir(exist_ok=True)
        self.cfg["metrics_dir"] = working_area / "metrics"
        self.cfg["metrics_dir"].mkdir(exist_ok=True)

        if self.cfg["game"] == "tictactoe":
            self.game = TicTacToe()
        elif re.match("hex[0-9]+", self.cfg["game"]):
            size = int(re.findall("hex([0-9]+)", self.cfg["game"])[0])
            self.game = Hex(size)
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
        self.base_model_path = str(base_model_path)

        self.cfg["self_play"]["temperature_policy_str"] = temperature_policy_to_str(
            self.cfg["self_play"]["temperature_policy"])
        self.cfg["model_compare"]["temperature_policy_str"] = temperature_policy_to_str(
            self.cfg["model_compare"]["temperature_policy"])

        assert self.cfg["model_compare"]["switching_winning_threshold"] >= 0.5
        assert self.cfg["model_compare"]["warning_losing_threshold"] >= 0.5

        self.lr_scheduler = LearningRateScheduler(self.cfg)

    def run_training_loop(self, run_id=None):
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.run_id = run_id
        metrics_filename = os.path.join(
            self.cfg["metrics_dir"], f"{self.run_id}.txt")

        best_model = self.base_model_path
        latest_model = self.base_model_path

        logging.info("Starting training process with config:")
        for line in dictionary_to_str(self.cfg).splitlines():
            logging.info(line)
        logging.info("run ID:\t" + self.run_id)
        logging.info("base model:\t" + best_model)
        logging.info("metrics file:\t" + metrics_filename)

        self._compile_selfplay_exe()

        for iter_num in range(self.cfg["self_play"]["iterations"]):
            logging.info(f"Training iteration {iter_num}")
            self.metrics = {}

            # Generate training data using the best model
            self._self_play(best_model)

            # Train latest model from training data
            latest_model = self._train(latest_model, iter_num)

            # Compare latest model to the current best, and switch if better
            best_model = self._compare_models(best_model, latest_model)

            # Write iteration metrics
            with open(metrics_filename, "a") as metrics_file:
                metrics_file.write(json.dumps(self.metrics) + "\n")

    def _self_play(self, model_path):
        profile = "dev" if self.cfg["debug"] == "true" else "release"
        games_dir = os.path.join(self.cfg["games_dir"], self.run_id)
        summary_file = os.path.join(games_dir, "selfplay_summary.json")
        data_entries_dir = os.path.join(
            games_dir, datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

        self_play_start_time = time.time()
        subprocess.run(" ".join([
            "cargo", "run", "--profile", profile, "-q", "--bin",
            self.self_play_exec, "--",
            "--model1-path", model_path,
            "--model2-path", model_path,
            "--games-num", str(self.cfg["self_play"]["games_num"]),
            "--out-dir1", data_entries_dir,
            "--out-dir2", data_entries_dir,
            "--summary-file", summary_file,
            "--sim-num", str(self.cfg["mcts"]["sim_num"]),
            "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
            "--temperature-policy", self.cfg["self_play"]["temperature_policy_str"],
            "--prior-noise-alpha", str(self.cfg["mcts"]["prior_noise_alpha"]),
            "--prior-noise-epsilon", str(self.cfg["mcts"]
                                         ["prior_noise_epsilon"]),
            "--threads", str(self.cfg["self_play"]["threads"]),
            "--processing-unit", "CPU" if self.cfg["cpu"] else "GPU",
            "--cache-size", str(self.cfg["mcts"]["cache_size"])]),
            stderr=sys.stderr, stdout=sys.stdout, shell=True, check=True)
        self.metrics["self_play_duration"] = time.time() - self_play_start_time

        with open(summary_file, "r") as f:
            summary = json.load(f)
        self.metrics.update({
            "net_activations_count": summary["net_activations_count"],
            "net_run_duration_average_us": summary["net_run_duration_average_us"],
            "search_count": summary["search_count"],
            "search_duration_average_ms": summary["search_duration_average_ms"],
            "cache_hit_ratio": summary["cache_hit_ratio"],
        })

    def _train(self, model_path, iter_num):
        games_dir = os.path.join(self.cfg["games_dir"], self.run_id)

        logging.debug("Loading current model")
        model = self.game.load_model(model_path, self.net_type)

        logging.debug("Loading games by current model")
        parser = DataParser(self.game, games_dir, self.cfg)
        train_dataset = tf.data.Dataset.from_generator(
            parser.generator, output_types=(tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(parser.get_parse_func())
        train_dataset = train_dataset.batch(
            self.cfg["training"]["batch_size"], drop_remainder=True)
        train_dataset = train_dataset.map(
            parser.get_after_batch_reshape_func())
        train_dataset = train_dataset.prefetch(4)

        lr = self.lr_scheduler.get_lr(
            iter_num * self.cfg["training"]["iteration_data_entries"])
        logging.debug("Training with learning rate %f", lr)
        keras.backend.set_value(model.optimizer.learning_rate, lr)

        logging.info("Fitting new model...")
        training_start_time = time.time()
        history = model.fit(train_dataset, epochs=1, verbose=0).history
        self.metrics.update({
            "value_loss": history["value_head_loss"][0],
            "policy_loss": history["policy_head_loss"][0],
            "value_accuracy": history["value_head_value_head_accuracy"][0],
            "policy_accuracy": history["policy_head_policy_head_accuracy"][0],
            "training_duration": time.time() - training_start_time
        })

        logging.info("Value loss {:.4f}".format(self.metrics["value_loss"]))
        logging.info("Policy loss {:.4f}".format(self.metrics["policy_loss"]))
        logging.info("Value accuracy {:.4f}".format(
            self.metrics["value_accuracy"]))
        logging.info("Policy accuracy {:.4f}".format(
            self.metrics["policy_accuracy"]))

        return self._save_model(model)

    def _compare_models(self, best_model, latest_model):
        logging.info("Comparing latest model to best model...")
        tmp_dirpath = tempfile.mkdtemp()
        try:
            compare_res_file = os.path.join(tmp_dirpath, "compare_result.json")
            tmp_games_dir = os.path.join(tmp_dirpath, "games")

            profile = "dev" if self.cfg["debug"] == "true" else "release"
            games_dir = os.path.join(self.cfg["games_dir"], self.run_id)
            data_entries_dir = os.path.join(
                games_dir, datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

            compare_start_time = time.time()
            subprocess.run([
                "cargo", "run", "--profile", profile, "-q", "--bin",
                self.self_play_exec, "--",
                "--model1-path", best_model,
                "--model2-path", latest_model,
                "--games-num", str(self.cfg["model_compare"]["games_num"]),
                # take the opportunity to generate more games to main games directory
                "--out-dir1", data_entries_dir,
                "--out-dir2", tmp_games_dir,
                "--summary-file", compare_res_file,
                "--sim-num", str(self.cfg["mcts"]["sim_num"]),
                "--explore-factor", str(self.cfg["mcts"]["explore_factor"]),
                "--temperature-policy", self.cfg["model_compare"]["temperature_policy_str"],
                "--prior-noise-alpha", str(self.cfg["mcts"]
                                           ["prior_noise_alpha"]),
                "--prior-noise-epsilon", str(self.cfg["mcts"]
                                             ["prior_noise_epsilon"]),
                "--threads", str(self.cfg["model_compare"]["threads"]),
                "--processing-unit", "CPU" if self.cfg["cpu"] else "GPU"],
                stderr=sys.stderr, stdout=sys.stdout, check=True)
            self.metrics["model_compare_duration"] = time.time() - \
                compare_start_time

            with open(compare_res_file, "r") as res_file:
                res = json.load(res_file)
            w1, w2, d = res["player1_wins"], res["player2_wins"], res["draws"]
            total_games = w1 + w2 + d
            winning, losing = w1 / total_games, w2 / total_games

            if winning > self.cfg["model_compare"]["switching_winning_threshold"]:
                best_model = latest_model
                # In case the new model is the new best model, take the opportunity and use the new games generated by
                # the comparison stage as training data in future training steps
                for filename in os.listdir(tmp_games_dir):
                    shutil.move(os.path.join(
                        tmp_games_dir, filename), data_entries_dir)
            elif losing > self.cfg["model_compare"]["warning_losing_threshold"]:
                logging.warn(
                    "New model is worse than previous one, losing ratio: %f", losing)

            logging.info("Best model: %s", best_model)
            return best_model
        finally:
            shutil.rmtree(tmp_dirpath)

    def _save_model(self, model):
        model_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + \
            "_{0:04x}".format(random.randint(0, 1 << 16))
        model_path = os.path.join(
            self.cfg["models_dir"], f"model_{model_time}")
        model.save(model_path, save_format='tf')
        return model_path

    def _compile_selfplay_exe(self):
        logging.info("Building Self-play executable...")
        profile = "dev" if self.cfg["debug"] == "true" else "release"
        subprocess.run([
            "cargo", "build",
            "--profile", profile, "-q",
            "--bin", self.self_play_exec],
            stderr=sys.stderr, stdout=sys.stdout, check=True)


class LearningRateScheduler:
    def __init__(self, cfg):
        cfg = cfg["training"]["learning_rate"]
        assert len(cfg) > 0

        thresholds = []
        for (idx, elm) in enumerate(cfg[:-1]):
            assert len(elm) == 2
            if idx > 0:
                # assert the steps thresholds are ordered
                assert elm[0] > cfg[idx - 1][0]
            thresholds.append((elm[0], elm[1]))
        self.thresholds = thresholds

        # last elm, no step threshold
        final_lr = cfg[-1]
        assert len(final_lr) == 1
        self.final_lr = final_lr[0]

    def get_lr(self, training_step):
        for (threshold, lr) in self.thresholds:
            if training_step < threshold:
                return lr
        return self.final_lr


def temperature_policy_to_str(temperature_policy):
    assert len(temperature_policy) > 0

    thresholds = []
    for (idx, elm) in enumerate(temperature_policy[:-1]):
        assert len(elm) == 2
        if idx > 0:
            # assert the steps thresholds are ordered
            assert elm[0] > temperature_policy[idx - 1][0]
        assert elm[1] >= 0  # valid temperature
        thresholds.append((elm[0], elm[1]))

    # last elm, no step threshold
    final_temp = temperature_policy[-1]
    assert len(final_temp) == 1
    final_temp = final_temp[0]

    if len(thresholds) == 0:
        return str(final_temp)
    else:
        thresholds = [f"{move_num},{temp}" for (move_num, temp) in thresholds]
        return f"{','.join(thresholds)},{final_temp}"
