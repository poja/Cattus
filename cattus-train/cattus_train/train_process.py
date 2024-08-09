import copy
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cattus_train.chess import Chess
from cattus_train.data_set import DataSet
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import Game

# For some reason, onnx.export is not thread-safe, so we need to lock it
ONNX_EXPORT_LOCK = threading.RLock()


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
        self._cfg: dict = copy.deepcopy(cfg)

        working_area = self._cfg["working_area"]
        working_area = working_area.format(CATTUS_TRAIN_TOP=os.getcwd(), GAME_NAME=self._cfg["game"])
        working_area = Path(working_area)
        self._cfg["working_area"] = working_area
        self._cfg["games_dir"] = working_area / "games"
        self._cfg["models_dir"] = working_area / "models"
        self._cfg["metrics_dir"] = working_area / "metrics"
        for path in (self._cfg[key] for key in ["working_area", "games_dir", "models_dir", "metrics_dir"]):
            os.makedirs(path, exist_ok=True)

        self._game: Game = None
        if self._cfg["game"] == "tictactoe":
            self._game = TicTacToe()
        elif re.match("hex[0-9]+", self._cfg["game"]):
            size = int(re.findall("hex([0-9]+)", self._cfg["game"])[0])
            self._game = Hex(size)
        elif self._cfg["game"] == "chess":
            self._game = Chess()
        else:
            raise ValueError("Unknown game argument in config file.")
        self._cfg["engine_path"] = Path(self._cfg["engine_path"])
        self._self_play_exec_name: str = "{}_self_play_runner".format(self._cfg["game"])
        self._self_play_exec_path: Path = None

        self._net_type: str = self._cfg["model"]["type"]
        base_model_path = self._cfg["model"]["base"]
        if base_model_path == "[none]":
            model = self._game.create_model(self._net_type, self._cfg)
            base_model_path = self._save_model(model)
        elif base_model_path == "[latest]":
            logging.warning("Choosing latest model based on directory name format")
            all_models = list(self._cfg["models_dir"].iterdir())
            if len(all_models) == 0:
                raise ValueError("Model [latest] was requested, but no existing models were found.")
            base_model_path = sorted(all_models)[-1]
        self._base_model_path: Path = base_model_path

        self._cfg["self_play"]["temperature_policy_str"] = temperature_policy_to_str(
            self._cfg["self_play"]["temperature_policy"]
        )
        self._cfg["model_compare"]["temperature_policy_str"] = temperature_policy_to_str(
            self._cfg["model_compare"]["temperature_policy"]
        )

        self._cfg["model_num"] = self._cfg.get("model_num", 1)
        assert self._cfg["model_compare"]["switching_winning_threshold"] >= 0.5
        assert self._cfg["model_compare"]["warning_losing_threshold"] >= 0.5

        if self._cfg["device"] == "auto":
            if torch.cuda.is_available():
                self._cfg["device"] = "cuda"
            elif torch.backends.mps.is_available():
                self._cfg["device"] = "mps"
            else:
                self._cfg["device"] = "cpu"
        assert self._cfg["device"] in ["cpu", "cuda", "mps"]

        self._lr_scheduler = LearningRateScheduler(self._cfg)

    def run_training_loop(self, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        self._run_id = run_id
        metrics_filename = self._cfg["metrics_dir"] / f"{self._run_id}.csv"

        best_model = (
            torch.load(self._base_model_path.with_suffix(".pt")),
            self._base_model_path,
        )
        latest_models = [best_model]
        if self._cfg["model_num"] > 1:
            for _ in range(self._cfg["model_num"] - 1):
                latest_models.append(copy.deepcopy(best_model))

        logging.info("Starting training process with config:")
        for line in dictionary_to_str(self._cfg).splitlines():
            logging.info(line)
        logging.info("run ID:\t%s", self._run_id)
        logging.info("base model:\t%s", self._base_model_path)
        logging.info("metrics file:\t%s", metrics_filename)

        self._compile_selfplay_exe()

        for iter_num in range(self._cfg["iterations"]):
            logging.info(f"Training iteration {iter_num}")
            self._metrics = {}

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

        games_dir = self._cfg["games_dir"] / self._run_id
        summary_file = games_dir / "selfplay_summary.json"
        data_entries_dir = games_dir / datetime.now().strftime("%y%m%d_%H%M%S_%f")

        self_play_start_time = time.time()
        subprocess.run(
            prepare_cmd(
                self._self_play_exec_path,
                "--model1-path",
                model_path,
                "--model2-path",
                model_path,
                "--games-num",
                self._cfg["self_play"]["games_num"],
                "--out-dir1",
                data_entries_dir,
                "--out-dir2",
                data_entries_dir,
                "--summary-file",
                summary_file,
                "--sim-num",
                self._cfg["mcts"]["sim_num"],
                "--explore-factor",
                self._cfg["mcts"]["explore_factor"],
                "--temperature-policy",
                self._cfg["self_play"]["temperature_policy_str"],
                "--prior-noise-alpha",
                self._cfg["mcts"]["prior_noise_alpha"],
                "--prior-noise-epsilon",
                self._cfg["mcts"]["prior_noise_epsilon"],
                "--threads",
                self._cfg["self_play"]["threads"],
                "--device",
                self._cfg["device"],
                "--cache-size",
                self._cfg["mcts"]["cache_size"],
            ),
            stderr=sys.stderr,
            stdout=sys.stdout,
            shell=True,
            check=True,
            cwd=self._cfg["engine_path"],
        )
        self._metrics["self_play_duration"] = time.time() - self_play_start_time

        with open(summary_file, "r") as f:
            summary = json.load(f)
        self._metrics.update(
            {
                "net_activations_count": summary["net_activations_count"],
                "net_run_duration_average_us": summary["net_run_duration_average_us"],
                "batch_size_average": summary["batch_size_average"],
                "search_count": summary["search_count"],
                "search_duration_average_ms": summary["search_duration_average_ms"],
                "cache_hit_ratio": summary["cache_hit_ratio"],
            }
        )

    def _train(self, models, iter_num) -> list[tuple[nn.Module, Path]]:
        train_data_dir = (
            self._cfg["games_dir"]
            if self._cfg["training"]["use_train_data_across_runs"]
            else self._cfg["games_dir"] / self._run_id
        )

        lr = self._lr_scheduler.get_lr(iter_num)
        logging.debug("Training models with learning rate %f", lr)

        losses = [None] * len(models)
        value_accuracies = [None] * len(models)
        policy_accuracies = [None] * len(models)
        training_durations = [None] * len(models)
        trained_models = [None] * len(models)

        def train_models(model_list: list[tuple[int, nn.Module]]):
            for m_idx, model in model_list:
                data_set = DataSet(self._game, train_data_dir, self._cfg, torch.device(self._cfg["device"]))
                data_loader = DataLoader(data_set, batch_size=self._cfg["training"]["batch_size"])

                def mask_illegal_moves(output, target):
                    output = torch.where(target >= 0, output, -1e10)
                    target = nn.ReLU()(target)
                    return output, target

                def loss_cross_entropy(output, target):
                    output, target = mask_illegal_moves(output, target)
                    return F.cross_entropy(output, target)

                def loss_fn(outputs, targets):
                    policy_output, value_output = outputs
                    policy_target, value_target = targets
                    policy_loss = loss_cross_entropy(policy_output, policy_target)
                    value_loss = F.mse_loss(value_output.squeeze(), value_target)
                    return policy_loss + value_loss

                model.train()
                model.to(self._cfg["device"])
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                final_batch = None
                training_start_time = time.time()
                for x, y in data_loader:
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    final_batch = (x, y)
                train_duration = time.time() - training_start_time
                model = model.to("cpu")

                def policy_head_accuracy(output, target):
                    output, target = mask_illegal_moves(output, target)
                    predicted = torch.argmax(output, dim=1)
                    target = torch.argmax(target, dim=1)
                    return (predicted == target).float().mean()

                def value_head_accuracy(output, target):
                    # Both the target and output should be in range [-1,1]
                    return 1 - torch.abs(target - output).mean() / 2

                with torch.no_grad():
                    model.eval()
                    final_x, final_y = final_batch
                    final_x = final_x.to("cpu")
                    final_y = (final_y[0].to("cpu"), final_y[1].to("cpu"))
                    final_outputs = model(final_x)
                    losses[m_idx] = loss_fn(final_outputs, final_y).detach().item()
                    policy_accuracies[m_idx] = policy_head_accuracy(final_outputs[0], final_y[0]).detach().item()
                    value_accuracies[m_idx] = value_head_accuracy(final_outputs[1], final_y[1]).detach().item()
                    training_durations[m_idx] = train_duration

                trained_models[m_idx] = (model, self._save_model(model))

        models = [(idx, model) for idx, (model, _model_path) in enumerate(models)]
        workers_num = min(self._cfg["training"].get("threads", 1), len(models))
        workers_num = workers_num if self._cfg["device"] == "cpu" else 1
        if workers_num > 1:
            # Divide the training into jobs
            jobs_per_cpu = len(models) // workers_num
            indices = np.arange(0, len(models) + jobs_per_cpu, jobs_per_cpu)
            jobs = [models[i:j] for i, j in zip(indices[:-1], indices[1:])]

            # Execute all jobs
            with ThreadPool(len(jobs)) as pool:
                pool.map(train_models, jobs)
        else:
            # Train on a single CPU thread
            train_models(models)

        for model_idx in range(len(models)):
            logging.info(f"Model{model_idx} training metrics:")
            logging.info("\tLoss            {:.4f}".format(losses[model_idx]))
            logging.info("\tValue accuracy  {:.4f}".format(value_accuracies[model_idx]))
            logging.info("\tPolicy accuracy {:.4f}".format(policy_accuracies[model_idx]))

            self._metrics.update(
                {
                    f"loss_{model_idx}": losses[model_idx],
                    f"value_accuracy_{model_idx}": value_accuracies[model_idx],
                    f"policy_accuracy_{model_idx}": policy_accuracies[model_idx],
                }
            )
        self._metrics["training_duration"] = sum(training_durations)

        return trained_models

    def _compare_models(self, best_model, latest_models) -> tuple[nn.Module, Path]:
        if self._cfg["model_compare"]["games_num"] == 0:
            assert len(latest_models) == 1, "Model comparison can be skipped only when one model is trained"
            logging.debug("Skipping model comparison, considering latest model as best")
            return latest_models[0]

        logging.info("Comparing latest models to best model...")
        for model_idx, (latest_model, latest_model_path) in enumerate(latest_models):
            # Compare the best model to the latest/trained model
            with tempfile.TemporaryDirectory() as tmp_dir:
                # take the opportunity to generate more games to main games directory
                games_dir = self._cfg["games_dir"] / self._run_id
                best_games_dir = games_dir / datetime.now().strftime("%y%m%d_%H%M%S_%f")
                trained_games_dir = Path(tmp_dir) / "games"

                compare_start_time = time.time()
                best_wr, trained_wr = self._compare_model_impl(
                    best_model[1], latest_model_path, best_games_dir, trained_games_dir
                )
                winning, losing = trained_wr, 1 - best_wr
                self._metrics["model_compare_duration"] = time.time() - compare_start_time
                self._metrics[f"trained_model_win_rate_{model_idx}"] = winning
                logging.debug(f"Trained model winning rate: {winning}")

                if winning > self._cfg["model_compare"]["switching_winning_threshold"]:
                    best_model = (latest_model, latest_model_path)
                    # In case the new model is the new best model, take the opportunity and use the new games generated
                    # by the comparison stage as training data in future training steps
                    for filename in os.listdir(trained_games_dir):
                        shutil.move(trained_games_dir / filename, best_games_dir)
                elif len(latest_models) == 1 and losing > self._cfg["model_compare"]["warning_losing_threshold"]:
                    logging.warn("New model is worse than previous one, losing ratio: %f", losing)
        return best_model

    def _compare_model_impl(self, model1_path: Path, model2_path: Path, model1_games_dir: Path, model2_games_dir: Path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            compare_res_file = Path(tmp_dir) / "compare_result.json"

            subprocess.run(
                prepare_cmd(
                    self._self_play_exec_path,
                    "--model1-path",
                    model1_path,
                    "--model2-path",
                    model2_path,
                    "--games-num",
                    self._cfg["model_compare"]["games_num"],
                    "--out-dir1",
                    model1_games_dir,
                    "--out-dir2",
                    model2_games_dir,
                    "--summary-file",
                    compare_res_file,
                    "--sim-num",
                    self._cfg["mcts"]["sim_num"],
                    "--explore-factor",
                    self._cfg["mcts"]["explore_factor"],
                    "--temperature-policy",
                    self._cfg["model_compare"]["temperature_policy_str"],
                    "--prior-noise-alpha",
                    self._cfg["mcts"]["prior_noise_alpha"],
                    "--prior-noise-epsilon",
                    self._cfg["mcts"]["prior_noise_epsilon"],
                    "--threads",
                    self._cfg["model_compare"]["threads"],
                    "--device",
                    self._cfg["device"],
                ),
                stderr=sys.stderr,
                stdout=sys.stdout,
                shell=True,
                check=True,
                cwd=self._cfg["engine_path"],
            )
            with open(compare_res_file, "r") as res_file:
                res = json.load(res_file)
            w1, w2, d = res["player1_wins"], res["player2_wins"], res["draws"]
            total_games = w1 + w2 + d
            return w1 / total_games, w2 / total_games

    def _save_model(self, model: nn.Module) -> Path:
        model_time = datetime.now().strftime("%y%m%d_%H%M%S_%f") + "_{0:04x}".format(random.randint(0, 1 << 16))
        model_path = self._cfg["models_dir"] / f"model_{model_time}"

        # Save model in Keras format
        torch.save(model, model_path.with_suffix(".pt"))

        # Save model in ONNX format
        model.eval()
        with torch.no_grad():
            sample_input = torch.randn(self._game.model_input_shape(self._net_type))
            with ONNX_EXPORT_LOCK:
                torch.onnx.export(
                    model,
                    sample_input,
                    model_path.with_suffix(".onnx"),
                    verbose=False,
                    input_names=["planes"],
                    output_names=["policy", "value"],
                    dynamic_axes={"planes": {0: "batch"}},  # TODO: consider removing this, may affect performance
                )

        return model_path

    def _compile_selfplay_exe(self):
        logging.info("Building Self-play executable...")
        profile = "dev" if self._cfg["debug"] else "release"
        subprocess.run(
            prepare_cmd(
                "cargo",
                "build",
                "--profile",
                profile,
                "-q",
                "--bin",
                self._self_play_exec_name,
            ),
            stderr=sys.stderr,
            stdout=sys.stdout,
            shell=True,
            check=True,
            cwd=self._cfg["engine_path"],
        )
        self._self_play_exec_path = (
            self._cfg["engine_path"]
            / "target"
            / ("debug" if self._cfg["debug"] else "release")
            / self._self_play_exec_name
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
        per_model_columns = [[f"{col}_{m_idx}" for col in per_model_columns] for m_idx in range(self._cfg["model_num"])]
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

        values = [str(self._metrics.get(metric, "")) for metric in columns]

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
        self._thresholds = thresholds

        # last elm, no iter threshold
        final_lr = cfg[-1]
        assert len(final_lr) == 1
        self.final_lr = final_lr[0]

    def get_lr(self, training_iter):
        for threshold, lr in self._thresholds:
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
