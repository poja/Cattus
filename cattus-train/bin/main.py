import argparse
import logging
import os
from pathlib import Path

import yaml

from cattus_train.train_process import TrainProcess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


CATTUS_ENGINE_TOP = Path(os.path.realpath(__file__)).parent.parent.parent / "cattus-engine"


def main():
    log_fmt = "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    # tensorflow is also using the logging lib, change it different from global
    logging.getLogger("tensorflow").setLevel(logging.WARN)

    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--run-id", type=str, help="Name of this run, default is current time")
    parser.add_argument("--logfile", type=str, default=None, help="All logs will be printed to file")
    args = parser.parse_args()

    if args.logfile is not None:
        fileHandler = logging.FileHandler(args.logfile)
        fileHandler.setFormatter(logging.Formatter(log_fmt))
        logging.getLogger().addHandler(fileHandler)

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)
    if "engine_path" not in config:
        config["engine_path"] = CATTUS_ENGINE_TOP

    tp = TrainProcess(config)
    tp.run_training_loop(run_id=args.run_id)


if __name__ == "__main__":
    main()
