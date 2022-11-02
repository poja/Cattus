import json
import logging
import os
from pathlib import Path
import shutil
import subprocess


REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_ttt_training")
CONFIG_FILE = os.path.join(TMP_DIR, "config.json")
PYTHON_MAIN = os.path.join(CATTUS_TOP, "train", "main.py")


def test_ttt_training():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[TicTactToe Training Test]: %(message)s')

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(f"""%YAML 1.2
---
game: "tictactoe"
working_area: {TMP_DIR}
mcts:
    sim_num: 1000
    explore_factor: 1.41421
    prior_noise_alpha: 0.0
    prior_noise_epsilon: 0.2
    cache_size: 1000
self_play:
    iterations: 3
    temperature_policy:
        - [30,    1.0]
        - [       0.0]
    games_num: 200
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
    compare:
        temperature_policy:
            - [       0.0]
        games_num: 4
        switching_winning_threshold: 0.55
        warning_losing_threshold: 0.55
        threads: 1
cpu: true
debug: false
""")

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
        logging.info("Training quality is sufficient")

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


def _get_metrics():
    path = Path(TMP_DIR) / 'metrics' / 'test'
    with path.open('r') as f:
        last_metric = f.readlines()[-1]
        return json.loads(last_metric)


if __name__ == "__main__":
    test_ttt_two_headed_net()