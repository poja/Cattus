#!/usr/bin/env python3

import os
import subprocess

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.join(TESTS_DIR, "..")

def run_cargo_tests():
    subprocess.check_call(["cargo", "test"], cwd=RL_TOP)

if __name__ == "__main__":
    run_cargo_tests()
