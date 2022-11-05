import json
import logging
import os
import math
import numpy as np
import shutil
import subprocess
import sys

from train.hex import Hex
from train.tictactoe import TicTacToe
from train.chess import Chess

REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_net_output")
MODEL_PATH = os.path.join(TMP_DIR, "model")
ENCODE_FILE = os.path.join(TMP_DIR, "encode_res.json")
OUTPUT_FILE = os.path.join(TMP_DIR, "output.json")

ASSERT_PYTHON_OUTPUT_EQ_REPEAT = 8
ASSERT_RUST_OUTPUT_EQ_REPEAT = 8


logging.basicConfig(
    level=logging.DEBUG,
    format='[Net Output Test]: %(message)s')


def is_outputs_equals(o1, o2):
    val1, probs1 = o1
    val2, probs2 = o2
    return math.isclose(val1, val2, rel_tol=1e-5) and \
        np.isclose(probs1, probs2, rtol=1e-3, atol=0).all()


def _test_net_output(game_name, game, positions):
    for position in positions:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR)

        try:
            model = create_model(game)

            # Encode position into tensor for Python model activation
            subprocess.run([
                "cargo", "run", "--profile", "dev", "-q", "--bin",
                "test_encode", "--",
                "--game", game_name,
                "--position", position,
                "--outfile", ENCODE_FILE],
                stderr=sys.stderr, stdout=sys.stdout, check=True)
            with open(ENCODE_FILE, "r") as encode_file:
                tensor_data = json.load(encode_file)
            shape = tuple(tensor_data["shape"])
            tensor = np.array(tensor_data["data"], dtype=np.float32)
            tensor = tensor.reshape(shape)
            assert shape == (1, game.BOARD_SIZE,
                             game.BOARD_SIZE, game.PLANES_NUM)

            # Run model from Python and assert all outputs are equal
            py_outputs = [model(tensor) for _ in range(ASSERT_PYTHON_OUTPUT_EQ_REPEAT)]
            py_outputs = [(val.numpy().item(), probs.numpy())
                          for (val, probs) in py_outputs]
            py_output = py_outputs[0]
            for out_other in py_outputs:
                assert is_outputs_equals(py_output, out_other)

            # Run model from Rust and assert all outputs are equal
            subprocess.run([
                "cargo", "run", "--profile", "dev", "-q", "--bin",
                "test_net_output", "--",
                "--game", game_name,
                "--position", position,
                "--model-path", MODEL_PATH,
                "--outfile", OUTPUT_FILE,
                "--repeat", str(ASSERT_RUST_OUTPUT_EQ_REPEAT)],
                stderr=sys.stderr, stdout=sys.stdout, check=True)
            with open(OUTPUT_FILE, "r") as output_file:
                output = json.load(output_file)
            rs_outputs = zip(output["vals"], output["probs"])
            rs_outputs = [(val, np.array(probs))
                          for (val, probs) in rs_outputs]
            rs_output = rs_outputs[0]
            for out_other in rs_outputs:
                assert is_outputs_equals(rs_output, out_other)

            # Assert Python output and Rust output are equal
            assert is_outputs_equals(py_output, rs_output)
        finally:
            if REMOVE_TMP_DIR_ON_FINISH:
                shutil.rmtree(TMP_DIR)


def create_model(game):
    cfg = {
        "model": {
            "residual_filter_num": 2,
            "residual_block_num": 2,
            "l2reg": 0.00005,
        },
        "cpu": True
    }
    model = game.create_model("ConvNetV1", cfg)
    model.save(MODEL_PATH, save_format='tf')
    return model


def test_ttt_net_output():
    _test_net_output(
        "tictactoe",
        TicTacToe(),
        [
            "___x__o_ox",
            "o_xx_x__ox",
            "o__xo__xxx",
            "o___x_o__x",
            "oo__x____x",
            "oo__o__oxx",
        ]
    )


def test_hex_net_output():
    _test_net_output(
        "hex",
        Hex(),
        [
            "reeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "eeeeeeeeeee"
            "r",
            "rererererer"
            "ererererere"
            "rererererer"
            "ererererere"
            "rererererer"
            "ererererere"
            "rererererer"
            "ererererere"
            "rererererer"
            "ererererere"
            "rererererer"
            "r",
            "reeeeeeeeee"
            "ereeeeeeeee"
            "eereeeeeeee"
            "eeereeeeeee"
            "eeeereeeeee"
            "eeeeereeeee"
            "eeeeeereeee"
            "eeeeeeereee"
            "eeeeeeeeree"
            "eeeeeeeeere"
            "eeeeeeeeeer"
            "r"
        ]
    )


def test_chess_net_output():
    _test_net_output(
        "chess",
        Chess(),
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "nnqrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNQRKBBR w - - 0 1",
            "8/1p6/3QR3/6k1/1P2b3/2P3K1/6b1/6r1/ w - - 0 1",
            "4k2r/6r1/8/8/8/8/3R4/R3K3 w Qk - 0 1"
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ]
    )


if __name__ == "__main__":
    test_ttt_net_output()
    test_hex_net_output()
    test_chess_net_output()
    logging.info("test passed")
