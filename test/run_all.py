#!/usr/bin/env python3

import os
import json
import subprocess

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILENAME = os.path.join(TESTS_DIR, "config.json")

def main(config):
    for test in config["tests"]:
        print()
        executable = os.path.join(TESTS_DIR, test["executable"])
        print("Executing '" + test["name"] + "' using ", executable)
        args = test["args"] if "args" in test else []
        subprocess.check_call(["python", executable] + args)
        print()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Trainer")
    # parser.add_argument("--config", type=str, required=True,
    #                     help="configuration file")
    # args = parser.parse_args()

    with open(CONFIG_FILENAME, "r") as config_file:
        config = json.load(config_file)
    main(config)
