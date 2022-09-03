#!/usr/bin/env python3
import json
import os
import sys
import shutil
import subprocess
import logging

DEBUG = True
REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "hex_scalar_net_test")
CONFIG_FILE = os.path.join(TMP_DIR, "config.json")
BIN_DIR = os.path.join(RL_TOP, "target", "debug" if DEBUG else "release")
SELF_PLAY_RUNNER = os.path.join(RL_TOP, "target", "debug", "{}_self_play_runner")
PYTHON_MAIN = os.path.join(RL_TOP, "train", "main.py")


def run_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Hex Scalar Net Test]: %(message)s')

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "game": "hex",
                "iterations": 2,
                "mcts_cfg": {
                    "sim_count": 100,
                    "explore_factor": 1.41421
                },
                "self_play_games_num": 3,
                "base_model": "[none]",
                "model_type": "scalar",
                "working_area": TMP_DIR,
                "self_play_exec": SELF_PLAY_RUNNER
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
