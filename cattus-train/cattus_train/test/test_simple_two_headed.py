import logging
import os
import shutil
import subprocess
import sys

REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_MAIN_BIN = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", "bin", "main.py"))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_simple_two_headed")
CONFIG_FILE = os.path.join(TMP_DIR, "config.yaml")


def _test_simple_two_headed(game_name):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(
                f"""%YAML 1.2
---
game: "{game_name}"
iterations: 3
cpu: true
debug: true
working_area: {TMP_DIR}
model:
    base: "[none]"
    type: "simple_two_headed"
mcts:
    sim_num: 10
    explore_factor: 1.41421
    prior_noise_alpha: 0.0
    prior_noise_epsilon: 0.2
    cache_size: 1000
self_play:
    temperature_policy:
        - [       0.0]
    games_num: 8
    threads: 1
training:
    latest_data_entries: 1024
    iteration_data_entries: 128
    batch_size: 4
    learning_rate:
        - [       0.001]
    l2reg: 0.00005
    use_train_data_across_runs: false
model_compare:
    temperature_policy:
        - [       0.0]
    games_num: 4
    switching_winning_threshold: 0.55
    warning_losing_threshold: 0.55
    threads: 1
"""
            )

        logging.info("Running self play and generating new models...")
        python = sys.executable
        subprocess.check_call(
            " ".join([python, TRAIN_MAIN_BIN, "--config", CONFIG_FILE]),
            env=dict(os.environ, PYTHONPATH=CATTUS_TOP),
            stderr=subprocess.STDOUT,
            shell=True,
        )

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


def test_ttt_two_headed():
    logging.basicConfig(
        level=logging.DEBUG, format="[TTT Simple Two Headed Test]: %(message)s"
    )
    _test_simple_two_headed("tictactoe")


def test_hex_two_headed():
    logging.basicConfig(
        level=logging.DEBUG, format="[Hex Simple Two Headed Test]: %(message)s"
    )
    for size in [4, 5, 7, 9, 11]:
        _test_simple_two_headed(f"hex{size}")


def test_chess_two_headed():
    logging.basicConfig(
        level=logging.DEBUG, format="[Chess Simple Two Headed Test]: %(message)s"
    )
    _test_simple_two_headed("chess")


if __name__ == "__main__":
    test_ttt_two_headed()
    test_hex_two_headed()
    test_chess_two_headed()
    print("test passed")
