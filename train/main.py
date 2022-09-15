#!/usr/bin/env python3

import argparse
import json
import logging
from train_process import TrainProcess

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    tp = TrainProcess(config)
    tp.run_training_loop()
