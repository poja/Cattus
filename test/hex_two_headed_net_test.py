#!/usr/bin/env python3
import json
import logging
import os
import shutil
import subprocess

DEBUG = True
REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "hex_two_headed_net_test")
CONFIG_FILE = os.path.join(TMP_DIR, "config.json")
BIN_DIR = os.path.join(RL_TOP, "target", "debug" if DEBUG else "release")
SELF_PLAY_RUNNER = "{}_self_play_runner"
PYTHON_MAIN = os.path.join(RL_TOP, "train", "main.py")


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
                    "sim_count": 100,
                    "explore_factor": 1.41421
                },
                "self_play": {
                    "iterations": 2,
                    "games_num": 3,
                },
                "model": {
                    "base": "[none]",
                    "type": "simple_two_headed",
                },
                "training": {
                    "entries_count": 10000
                },
                "debug": "true"
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
