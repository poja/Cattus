import logging
import os
import shutil
import subprocess
from pathlib import Path

REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_ttt_training")
CONFIG_FILE = os.path.join(TMP_DIR, "config.yaml")
PYTHON_MAIN = os.path.join(CATTUS_TOP, "train", "main.py")


def test_ttt_training():
    logging.basicConfig(
        level=logging.DEBUG, format="[TicTactToe Training Test]: %(message)s"
    )

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(
                f"""%YAML 1.2
---
game: "tictactoe"
iterations: 3
cpu: true
debug: false
working_area: {TMP_DIR}
model:
    base: "[none]"
    type: "ConvNetV1"
    residual_block_num: 5
    residual_filter_num: 8
    value_head_conv_output_channels_num: 8
    policy_head_conv_output_channels_num: 8
mcts:
    sim_num: 600
    explore_factor: 1.41421
    prior_noise_alpha: 0.0
    prior_noise_epsilon: 0.2
    cache_size: 1000000
self_play:
    temperature_policy:
        - [5,   1.0]
        - [     0.0]
    games_num: 100
    threads: 8
training:
    latest_data_entries: 3000
    iteration_data_entries: 3000
    batch_size: 16
    learning_rate:
        - [       0.001]
    l2reg: 0.00005
    use_train_data_across_runs: false
model_compare:
    temperature_policy:
        - [       0.0]
    games_num: 0
    switching_winning_threshold: 0.55
    warning_losing_threshold: 0.55
    threads: 8
"""
            )

        logging.info("Running self play and generating new models...")
        subprocess.check_call(
            ["python", PYTHON_MAIN, "--config", CONFIG_FILE, "--run-id", "test"],
            stderr=subprocess.STDOUT,
        )

        logging.info("Checking quality of training...")
        metrics = _get_metrics()
        assert float(metrics["value_loss_0"]) > 0
        assert float(metrics["policy_loss_0"]) > 0
        assert float(metrics["value_accuracy_0"]) > 0.6
        assert float(metrics["policy_accuracy_0"]) > 0.4
        logging.info("Training quality is sufficient")

    finally:
        if REMOVE_TMP_DIR_ON_FINISH:
            shutil.rmtree(TMP_DIR)


def _get_metrics():
    path = Path(TMP_DIR) / "metrics" / "test.csv"
    with path.open("r") as f:
        lines = f.readlines()
        columns = lines[0].split(",")
        last_metric = lines[-1].split(",")
        metrics = {}
        for i in range(len(columns)):
            metrics[columns[i]] = last_metric[i]
        return metrics


if __name__ == "__main__":
    test_ttt_training()
