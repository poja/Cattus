#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess


REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_tictactoe_two_headed_net")
CONFIG_FILE = os.path.join(TMP_DIR, "config.yaml")
PYTHON_MAIN = os.path.join(RL_TOP, "train", "main.py")


def run_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[TicTactToe Two Headed Net Test]: %(message)s')

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "game": "tictactoe",
                "working_area": TMP_DIR,
                "mcts": {
                    "sim_num": 1000,
                    "explore_factor": 1.41421,
                    "cache_size": 1000,
                    "prior_noise_alpha": 0.0,
                    "prior_noise_epsilon": 0.0,
                },
                "self_play": {
                    "iterations": 3,
                    "temperature_policy": [
                        [30,    1.0],
                        [       0.0]
                    ],
                    "games_num": 200,
                    "threads": 1,
                },
                "model": {
                    "base": "[none]",
                    "type": "simple_two_headed",
                    "l2reg": 0.00005,
                },
                "training": {
                    "latest_data_entries": 1024,
                    "iteration_data_entries": 128,
                    "batch_size": 4,
                    "learning_rate": [
                        [0.001]
                    ],
                    "compare": {
                        "temperature_policy": [
                            [       0.0]
                        ],
                        "games_num": 4,
                        "switching_winning_threshold": 0.55,
                        "warning_losing_threshold": 0.55,
                        "threads": 1,
                    }
                },
                "cpu": True,
                "debug": True,
            }, f)

        logging.info("Running self play and generating new models...")
        subprocess.check_call([
            "python", PYTHON_MAIN,
            "--config", CONFIG_FILE,
            "--run-id", "test"],
            stderr=subprocess.STDOUT)

        logging.info("Checking quality of training...")
        metrics = _get_metrics()
        assert metrics['value_loss'] > 0
        assert metrics['policy_loss'] > 0
        assert metrics['value_accuracy'] > 0.6
        assert metrics['policy_accuracy'] > 0.2

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


def _get_metrics():
    path = Path(TMP_DIR) / 'metrics' / 'test'
    with path.open('r') as f:
        lines = f.read().strip().split()
        return json.loads(lines[-1])


if __name__ == "__main__":
    run_test()
