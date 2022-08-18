#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess

DEBUG = True
REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "hex_scalar_net_test")
MODEL_SCRIPT = os.path.join(RL_TOP, "train", "net",
                            "hex", "scalar_value_net.py")
BIN_DIR = os.path.join(RL_TOP, "target", "debug" if DEBUG else "release")
SELF_PLAY_RUNNER = "hex_self_play_runner"


def test_print(*args):
    print("[Hex Scalar Net Test]", *args)


def run_test():
    shutil.rmtree(TMP_DIR)

    model_path1 = os.path.join(TMP_DIR, "model1")
    model_path2 = os.path.join(TMP_DIR, "model2")
    self_play_dir1 = os.path.join(TMP_DIR, "self_play1")
    self_play_dir2 = os.path.join(TMP_DIR, "self_play2")

    try:
        test_print("creating a new model at", model_path1)
        subprocess.check_call([
            "python", MODEL_SCRIPT,
            "--create",
            "--out", model_path1],
            stderr=subprocess.STDOUT)

        test_print("running self play using the model...")
        subprocess.check_call([
            "cargo", "run", "--bin",
            SELF_PLAY_RUNNER, "--",
            "--model", model_path1,
            "--games-num", "10",
            "--out-dir", self_play_dir1,
            "--sim-count", "100"],
            stderr=subprocess.STDOUT)

        test_print(
            "training model on self play data and saving new model at", model_path2)
        subprocess.check_call([
            "python", MODEL_SCRIPT,
            "--train",
            "--model", model_path1,
            "--data", self_play_dir1,
            "--out", model_path2],
            stderr=subprocess.STDOUT)

        test_print("running self play using trained model...")
        subprocess.check_call([
            "cargo", "run", "--bin",
            SELF_PLAY_RUNNER, "--",
            "--model", model_path2,
            "--games-num", "10",
            "--out-dir", self_play_dir2,
            "--sim-count", "100"],
            stderr=subprocess.STDOUT)
    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    run_test()
