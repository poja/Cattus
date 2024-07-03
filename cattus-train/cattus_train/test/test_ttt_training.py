import logging
import os
import subprocess
import tempfile
from pathlib import Path

REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_MAIN_BIN = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", "bin", "main.py"))


def test_ttt_training():
    logging.basicConfig(
        level=logging.DEBUG, format="[TicTactToe Training Test]: %(message)s"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = os.path.join(tmp_dir, "config.yaml")

        with open(config_file, "w") as f:
            f.write(
                f"""%YAML 1.2
---
game: "tictactoe"
iterations: 3
device: cpu
debug: false
working_area: {tmp_dir}
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
            ["python", TRAIN_MAIN_BIN, "--config", config_file, "--run-id", "test"],
            stderr=subprocess.STDOUT,
        )

        logging.info("Checking quality of training...")
        metrics = _get_metrics(tmp_dir)
        assert float(metrics["loss_0"]) > 0
        assert float(metrics["value_accuracy_0"]) > 0.6
        assert float(metrics["policy_accuracy_0"]) > 0.4
        logging.info("Training quality is sufficient")


def _get_metrics(working_area):
    path = Path(working_area) / "metrics" / "test.csv"
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
