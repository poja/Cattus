#!/usr/bin/env python3

import json
import logging
import os
import shutil
import subprocess


DEBUG = True
REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_hex_two_headed_net")
CONFIG_FILE = os.path.join(TMP_DIR, "config.json")
PYTHON_MAIN = os.path.join(CATTUS_TOP, "train", "main.py")


def run_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Hex Two Headed Net Test]: %(message)s')

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "game": "hex",
                "working_area": TMP_DIR,
                "mcts": {
                    "sim_num": 10,
                    "explore_factor": 1.41421,
                    "cache_size": 1000,
                    "prior_noise_alpha": 0.0,
                    "prior_noise_epsilon": 0.0,
                },
                "self_play": {
                    "iterations": 2,
                    "temperature_policy": [
                        [       0.0]
                    ],
                    "games_num": 8,
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
            "--config", CONFIG_FILE],
            stderr=subprocess.STDOUT)

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    run_test()
