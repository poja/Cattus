import logging
import os
import shutil
import subprocess
import sys

REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_simple_two_headed")
CONFIG_FILE = os.path.join(TMP_DIR, "config.yaml")


def _test_simple_two_headed(game_name):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(f"""%YAML 1.2
---
game: "{game_name}"
working_area: {TMP_DIR}
mcts:
    sim_num: 10
    explore_factor: 1.41421
    prior_noise_alpha: 0.0
    prior_noise_epsilon: 0.2
    cache_size: 1000
self_play:
    iterations: 3
    temperature_policy:
        - [       0.0]
    games_num: 8
    threads: 1
model:
    base: "[none]"
    type: "simple_two_headed"
    l2reg: 0.00005
training:
    latest_data_entries: 1024
    iteration_data_entries: 128
    batch_size: 4
    learning_rate:
        - [       0.001]
model_compare:
    temperature_policy:
        - [       0.0]
    games_num: 4
    switching_winning_threshold: 0.55
    warning_losing_threshold: 0.55
    threads: 1
cpu: true
debug: true
""")

        logging.info("Running self play and generating new models...")
        python = sys.executable
        subprocess.check_call(" ".join(
            [python, "-m", "train.main", "--config", CONFIG_FILE]),
            env=dict(os.environ, PYTHONPATH=CATTUS_TOP),
            stderr=subprocess.STDOUT,
            shell=True)

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


def test_ttt_two_headed():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[TTT Simple Two Headed Test]: %(message)s')
    _test_simple_two_headed("tictactoe")


def test_hex_two_headed():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Hex Simple Two Headed Test]: %(message)s')
    _test_simple_two_headed("hex5")
    _test_simple_two_headed("hex7")
    _test_simple_two_headed("hex9")
    _test_simple_two_headed("hex11")


def test_chess_two_headed():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Chess Simple Two Headed Test]: %(message)s')
    _test_simple_two_headed("chess")


if __name__ == "__main__":
    test_ttt_two_headed()
    test_hex_two_headed()
    test_chess_two_headed()
    print("test passed")
