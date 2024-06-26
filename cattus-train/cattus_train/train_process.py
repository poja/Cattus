import copy
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import keras
import onnx
import tf2onnx

from cattus_train.chess import Chess
from cattus_train.data_parser import DataParser
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe

CATTUS_TRAIN_TOP = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)


def dictionary_to_str(d, indent=0):
    s = ""
    for key, value in d.items():
        if isinstance(value, dict):
            s += "  " * indent + str(key) + ":\n"
            s += dictionary_to_str(value, indent + 1)
        else:
            s += "  " * indent + str(key) + ": \t" + str(value) + "\n"
    return s


def prepare_cmd(*args):
    if None in args:
        raise ValueError("CMD error: None at index {}".format(args.index(None)))
    return " ".join([str(arg) for arg in args])


class TrainProcess:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)

        working_area = self.cfg["working_area"]
        working_area = working_area.format(
            CATTUS_TRAIN_TOP=CATTUS_TRAIN_TOP, GAME_NAME=self.cfg["game"]
        )
        working_area = Path(working_area)
        self.cfg["working_area"] = working_area
        self.cfg["games_dir"] = working_area / "games"
        self.cfg["models_dir"] = working_area / "models"
        self.cfg["metrics_dir"] = working_area / "metrics"
        for path in (
            self.cfg[key]
            for key in ["working_area", "games_dir", "models_dir", "metrics_dir"]
        ):
            os.makedirs(path, exist_ok=True)

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
            assert model.get_layer("value_head").output.shape == (None, 1)
            assert model.get_layer("policy_head").output.shape == (
                None,
                self.game.MOVE_NUM,
            )
            base_model_path = self._save_model(model)
        elif base_model_path == "[latest]":
            logging.warning("Choosing latest model based on directory name format")
            all_models = list(self.cfg["models_dir"].iterdir())
            if len(all_models) == 0:
                raise ValueError(
                    "Model [latest] was requested, but no existing models were found."
                )
            base_model_path = sorted(all_models)[-1]
        self.base_model_path: Path = base_model_path

        self.cfg["self_play"]["temperature_policy_str"] = temperature_policy_to_str(
            self.cfg["self_play"]["temperature_policy"]
        )
        self.cfg["model_compare"]["temperature_policy_str"] = temperature_policy_to_str(
            self.cfg["model_compare"]["temperature_policy"]
        )

        self.cfg["model_num"] = self.cfg.get("model_num", 1)
        assert self.cfg["model_compare"]["switching_winning_threshold"] >= 0.5
        assert self.cfg["model_compare"]["warning_losing_threshold"] >= 0.5

        self.lr_scheduler = LearningRateScheduler(self.cfg)

    def run_training_loop(self, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        self.run_id = run_id
        metrics_filename = os.path.join(self.cfg["metrics_dir"], f"{self.run_id}.csv")

        best_model = (
            self.game.load_model(self.base_model_path, self.net_type),
            self.base_model_path,
        )
        latest_models = [best_model]
        if self.cfg["model_num"] > 1:
            for _ in range(self.cfg["model_num"] - 1):
                model = self.game.create_model(self.net_type, self.cfg)
                latest_models.append((model, self._save_model(model)))

        logging.info("Starting training process with config:")
        for line in dictionary_to_str(self.cfg).splitlines():
            logging.info(line)
        logging.info("run ID:\t" + self.run_id)
        logging.info("base model:\t" + str(self.base_model_path))
        logging.info("metrics file:\t" + metrics_filename)

        self._compile_selfplay_exe()

        for iter_num in range(self.cfg["iterations"]):
            logging.info(f"Training iteration {iter_num}")
            self.metrics = {}

            # Generate training data using the best model
            self._self_play(best_model[1])

            # Train latest models from training data
            latest_models = self._train(latest_models, iter_num)

            # Compare latest model to the current best, and switch if better
            best_model = self._compare_models(best_model, latest_models)

            # Write iteration metrics
            self._write_metrics(metrics_filename)

    def _self_play(self, model_path: Path):
        logging.info("Self playing using model: %s", model_path)

        profile = "dev" if self.cfg["debug"] else "release"
        games_dir = os.path.join(self.cfg["games_dir"], self.run_id)
        summary_file = os.path.join(games_dir, "selfplay_summary.json")
        data_entries_dir = os.path.join(
            games_dir, datetime.now().strftime("%y%m%d_%H%M%S_%f")
        )

        self_play_start_time = time.time()
        subprocess.run(
            prepare_cmd(
                "cargo",
                "run",
                "--profile",
                profile,
                "-q",
                "--bin",
                self.self_play_exec,
                "--",
                "--model1-path",
                model_path.with_suffix(".onnx"),
                "--model2-path",
                model_path.with_suffix(".onnx"),
                "--games-num",
                self.cfg["self_play"]["games_num"],
                "--out-dir1",
                data_entries_dir,
                "--out-dir2",
                data_entries_dir,
                "--summary-file",
                summary_file,
                "--sim-num",
                self.cfg["mcts"]["sim_num"],
                "--explore-factor",
                self.cfg["mcts"]["explore_factor"],
                "--temperature-policy",
                self.cfg["self_play"]["temperature_policy_str"],
                "--prior-noise-alpha",
                self.cfg["mcts"]["prior_noise_alpha"],
                "--prior-noise-epsilon",
                self.cfg["mcts"]["prior_noise_epsilon"],
                "--threads",
                self.cfg["self_play"]["threads"],
                "--processing-unit",
                "CPU" if self.cfg["cpu"] else "GPU",
                "--cache-size",
                self.cfg["mcts"]["cache_size"],
            ),
            stderr=sys.stderr,
            stdout=sys.stdout,
            shell=True,
            check=True,
            cwd=self.cfg["engine_path"],
        )
        self.metrics["self_play_duration"] = time.time() - self_play_start_time

        with open(summary_file, "r") as f:
            summary = json.load(f)
        self.metrics.update(
            {
                "net_activations_count": summary["net_activations_count"],
                "net_run_duration_average_us": summary["net_run_duration_average_us"],
                "batch_size_average": summary["batch_size_average"],
                "search_count": summary["search_count"],
                "search_duration_average_ms": summary["search_duration_average_ms"],
                "cache_hit_ratio": summary["cache_hit_ratio"],
            }
        )

    def _train(self, models, iter_num) -> list[tuple[keras.Model, Path]]:
        train_data_dir = (
            self.cfg["games_dir"]
            if self.cfg["training"]["use_train_data_across_runs"]
            else os.path.join(self.cfg["games_dir"], self.run_id)
        )

        lr = self.lr_scheduler.get_lr(iter_num)
        logging.debug("Training models with learning rate %f", lr)

        # values_loss = [None] * len(models)
        # values_loss = [None] * len(models)
        loss = [None] * len(models)
        value_accuracy = [None] * len(models)
        policy_accuracy = [None] * len(models)
        training_duration = [None] * len(models)
        trained_models = [None] * len(models)

        def train_models(model_list):
            for model_idx, (model, _model_path) in model_list:
                parser = DataParser(self.game, train_data_dir, self.cfg)
                train_dataset = parser.create_train_dataset()

                # TODO: not sure if this is the right way to set the learning rate
                # tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
                model.optimizer.learning_rate = lr

                training_start_time = time.time()
                history = model.fit(train_dataset, epochs=1, verbose=0).history
                train_dur = time.time() - training_start_time

                # values_loss[model_idx] = history["value_head_loss"][0]
                # policy_loss[model_idx] = history["policy_head_loss"][0]
                loss[model_idx] = history["loss"][0]
                value_accuracy[model_idx] = history["value_head_value_head_accuracy"][0]
                policy_accuracy[model_idx] = history[
                    "policy_head_policy_head_accuracy"
                ][0]
                training_duration[model_idx] = train_dur

                trained_models[model_idx] = (model, self._save_model(model))

        # Divide the training into jobs
        models = list(enumerate(models))
        workers_num = min(self.cfg["training"].get("threads", 1), len(models))
        jobs = []
        for idx in range(workers_num):
            i = idx / workers_num * len(models)
            j = (idx + 1) / workers_num * len(models)
            jobs.append(models[int(i) : int(j)])

        # Execute all jobs
        with multiprocessing.pool.ThreadPool(len(jobs)) as pool:
            pool.map(train_models, jobs)

        for model_idx in range(len(models)):
            logging.info(f"Model{model_idx} training metrics:")
            # logging.info("\tValue loss      {:.4f}".format(values_loss[model_idx]))
            # logging.info("\tPolicy loss     {:.4f}".format(policy_loss[model_idx]))
            logging.info("\tLoss            {:.4f}".format(loss[model_idx]))
            logging.info("\tValue accuracy  {:.4f}".format(value_accuracy[model_idx]))
            logging.info("\tPolicy accuracy {:.4f}".format(policy_accuracy[model_idx]))

            self.metrics.update(
                {
                    f"loss_{model_idx}": loss[model_idx],
                    f"value_accuracy_{model_idx}": value_accuracy[model_idx],
                    f"policy_accuracy_{model_idx}": policy_accuracy[model_idx],
                }
            )
        self.metrics["training_duration"] = sum(training_duration)

        return trained_models

    def _compare_models(self, best_model, latest_models) -> tuple[keras.Model, Path]:
        if self.cfg["model_compare"]["games_num"] == 0:
            assert (
                len(latest_models) == 1
            ), "Model comparison can be skipped only when one model is trained"
            logging.debug("Skipping model comparison, considering latest model as best")
            return latest_models[0]

        logging.info("Comparing latest models to best model...")
        for model_idx, (latest_model, latest_model_path) in enumerate(latest_models):
            # Compare the best model to the latest/trained model
            with tempfile.TemporaryDirectory() as tmp_dir:
                # take the opportunity to generate more games to main games directory
                games_dir = Path(self.cfg["games_dir"]) / self.run_id
                best_games_dir = games_dir / datetime.now().strftime("%y%m%d_%H%M%S_%f")
                trained_games_dir = Path(tmp_dir) / "games"

                compare_start_time = time.time()
                best_wr, trained_wr = self._compare_model_impl(
                    best_model[1], latest_model_path, best_games_dir, trained_games_dir
                )
                winning, losing = trained_wr, 1 - best_wr
                self.metrics["model_compare_duration"] = (
                    time.time() - compare_start_time
                )
                self.metrics[f"trained_model_win_rate_{model_idx}"] = winning
                logging.debug(f"Trained model winning rate: {winning}")

                if winning > self.cfg["model_compare"]["switching_winning_threshold"]:
                    best_model = (latest_model, latest_model_path)
                    # In case the new model is the new best model, take the opportunity and use the new games generated
                    # by the comparison stage as training data in future training steps
                    for filename in os.listdir(trained_games_dir):
                        shutil.move(trained_games_dir / filename, best_games_dir)
                elif (
                    len(latest_models) == 1
                    and losing > self.cfg["model_compare"]["warning_losing_threshold"]
                ):
                    logging.warn(
                        "New model is worse than previous one, losing ratio: %f", losing
                    )
        return best_model

    def _compare_model_impl(
        self, model1_path: Path, model2_path: Path, model1_games_dir, model2_games_dir
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            compare_res_file = os.path.join(tmp_dir, "compare_result.json")
            profile = "dev" if self.cfg["debug"] else "release"

            subprocess.run(
                prepare_cmd(
                    "cargo",
                    "run",
                    "--profile",
                    profile,
                    "-q",
                    "--bin",
                    self.self_play_exec,
                    "--",
                    "--model1-path",
                    model1_path.with_suffix(".onnx"),
                    "--model2-path",
                    model2_path.with_suffix(".onnx"),
                    "--games-num",
                    self.cfg["model_compare"]["games_num"],
                    "--out-dir1",
                    model1_games_dir,
                    "--out-dir2",
                    model2_games_dir,
                    "--summary-file",
                    compare_res_file,
                    "--sim-num",
                    self.cfg["mcts"]["sim_num"],
                    "--explore-factor",
                    self.cfg["mcts"]["explore_factor"],
                    "--temperature-policy",
                    self.cfg["model_compare"]["temperature_policy_str"],
                    "--prior-noise-alpha",
                    self.cfg["mcts"]["prior_noise_alpha"],
                    "--prior-noise-epsilon",
                    self.cfg["mcts"]["prior_noise_epsilon"],
                    "--threads",
                    self.cfg["model_compare"]["threads"],
                    "--processing-unit",
                    "CPU" if self.cfg["cpu"] else "GPU",
                ),
                stderr=sys.stderr,
                stdout=sys.stdout,
                shell=True,
                check=True,
                cwd=self.cfg["engine_path"],
            )
            with open(compare_res_file, "r") as res_file:
                res = json.load(res_file)
            w1, w2, d = res["player1_wins"], res["player2_wins"], res["draws"]
            total_games = w1 + w2 + d
            return w1 / total_games, w2 / total_games

    def _save_model(self, model: keras.Model) -> Path:
        model_time = datetime.now().strftime("%y%m%d_%H%M%S_%f") + "_{0:04x}".format(
            random.randint(0, 1 << 16)
        )

        model_path = Path(self.cfg["models_dir"]) / f"model_{model_time}"

        # Save model in Keras format
        model.save(model_path.with_suffix(".keras"))

        # Save model in ONNX format
        input_signature = self.game.model_input_signature(self.net_type, self.cfg)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
        onnx.save(onnx_model, model_path.with_suffix(".onnx"))

        return model_path

    def _compile_selfplay_exe(self):
        logging.info("Building Self-play executable...")
        profile = "dev" if self.cfg["debug"] else "release"
        subprocess.run(
            prepare_cmd(
                "cargo",
                "build",
                "--profile",
                profile,
                "-q",
                "--bin",
                self.self_play_exec,
            ),
            stderr=sys.stderr,
            stdout=sys.stdout,
            shell=True,
            check=True,
            cwd=self.cfg["engine_path"],
        )

    def _write_metrics(self, filename):
        per_model_columns = [
            # "value_loss",
            # "policy_loss",
            "loss",
            "value_accuracy",
            "policy_accuracy",
            "trained_model_win_rate",
        ]
        per_model_columns = [
            [f"{col}_{m_idx}" for col in per_model_columns]
            for m_idx in range(self.cfg["model_num"])
        ]
        columns = [
            "net_run_duration_average_us",
            "batch_size_average",
            "net_activations_count",
            "search_duration_average_ms",
            "search_count",
            "cache_hit_ratio",
            "self_play_duration",
            "training_duration",
            "model_compare_duration",
        ] + sum(per_model_columns, [])

        values = [str(self.metrics.get(metric, "")) for metric in columns]

        # write columns
        if not os.path.exists(filename):
            with open(filename, "w") as metrics_file:
                metrics_file.write(",".join(columns) + "\n")

        # write values
        with open(filename, "a") as metrics_file:
            metrics_file.write(",".join(values) + "\n")


class LearningRateScheduler:
    def __init__(self, cfg):
        cfg = cfg["training"]["learning_rate"]
        assert len(cfg) > 0

        thresholds = []
        for idx, elm in enumerate(cfg[:-1]):
            assert len(elm) == 2
            if idx > 0:
                # assert the iters thresholds are ordered
                assert elm[0] > cfg[idx - 1][0]
            thresholds.append((elm[0], elm[1]))
        self.thresholds = thresholds

        # last elm, no iter threshold
        final_lr = cfg[-1]
        assert len(final_lr) == 1
        self.final_lr = final_lr[0]

    def get_lr(self, training_iter):
        for threshold, lr in self.thresholds:
            if training_iter < threshold:
                return lr
        return self.final_lr


def temperature_policy_to_str(temperature_policy):
    assert len(temperature_policy) > 0

    thresholds = []
    for idx, elm in enumerate(temperature_policy[:-1]):
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
