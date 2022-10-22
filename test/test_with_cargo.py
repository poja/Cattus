#!/usr/bin/env python3

import os
import subprocess

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))

def test_run_all_cargo_tests():
    subprocess.check_call(["cargo", "test"], cwd=CATTUS_TOP)

if __name__ == "__main__":
    run_cargo_tests()
