#!/usr/bin/env python3

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
MODEL_SCRIPT = os.path.join(
    RL_TOP, "train", "net", "hex", "create_net.py")
BIN_DIR = os.path.join(RL_TOP, "target", "debug" if DEBUG else "release")
SELF_PLAY_RUNNER = "hex_self_play_runner"


def run_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Hex Scalar Net Test]: %(message)s')

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    model_path1 = os.path.join(TMP_DIR, "model1")
    model_path2 = os.path.join(TMP_DIR, "model2")
    self_play_dir1 = os.path.join(TMP_DIR, "self_play1")
    self_play_dir2 = os.path.join(TMP_DIR, "self_play2")

    try:
        logging.info("creating a new model at %s", model_path1)
        subprocess.check_call([
            "python", MODEL_SCRIPT,
            "--type", "simple_scalar",
            "--create",
            "--out-dir", model_path1],
            stderr=subprocess.STDOUT)

        logging.info("running self play using the model...")
        subprocess.check_call([
            "cargo", "run", "--bin",
            SELF_PLAY_RUNNER, "--",
            "--net-type", "scalar_net",
            "--model-path", model_path1,
            "--games-num", "10",
            "--out-dir", self_play_dir1,
            "--sim-count", "100"],
            stderr=subprocess.STDOUT)

        logging.info(
            "training model on self play data and saving new model at %s", model_path2)
        subprocess.check_call([
            "python", MODEL_SCRIPT,
            "--type", "simple_scalar",
            "--train",
            "--model-path", model_path1,
            "--data-dir", self_play_dir1,
            "--out-dir", model_path2],
            stderr=subprocess.STDOUT)

        logging.info("running self play using trained model...")
        subprocess.check_call([
            "cargo", "run", "--bin",
            SELF_PLAY_RUNNER, "--",
            "--net-type", "scalar_net",
            "--model-path", model_path2,
            "--games-num", "10",
            "--out-dir", self_play_dir2,
            "--sim-count", "100"],
            stderr=subprocess.STDOUT)
    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    run_test()
