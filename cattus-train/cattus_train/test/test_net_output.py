import json
import logging
import math
import os
import subprocess
import sys
import tempfile

import keras
import numpy as np
import onnx
import tf2onnx

from cattus_train.chess import Chess
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import TrainableGame

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_ENGINE_TOP = os.path.abspath(
    os.path.join(TESTS_DIR, "..", "..", "..", "cattus-engine")
)

ASSERT_PYTHON_OUTPUT_EQ_REPEAT = 8
ASSERT_RUST_OUTPUT_EQ_REPEAT = 8


logging.basicConfig(level=logging.DEBUG, format="[Net Output Test]: %(message)s")


def is_outputs_equals(o1, o2):
    val1, probs1 = o1
    val2, probs2 = o2
    return (
        math.isclose(val1, val2, rel_tol=1e-5, abs_tol=1e-6)
        and np.isclose(probs1, probs2, rtol=1e-3, atol=1e-6).all()
    )


def _test_net_output(game_name: str, game: TrainableGame, positions):
    for position in positions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.onnx")
            encode_path = os.path.join(tmp_dir, "encode_res.json")
            output_file = os.path.join(tmp_dir, "output.json")

            model = create_model(game, model_path)

            # Encode position into tensor for Python model activation
            subprocess.run(
                [
                    "cargo",
                    "run",
                    "--profile",
                    "dev",
                    "-q",
                    "--bin",
                    "test_encode",
                    "--",
                    "--game",
                    game_name,
                    "--position",
                    position,
                    "--outfile",
                    encode_path,
                ],
                stderr=sys.stderr,
                stdout=sys.stdout,
                check=True,
                cwd=CATTUS_ENGINE_TOP,
            )
            with open(encode_path, "r") as encode_file:
                tensor_data = json.load(encode_file)
            shape = tuple(tensor_data["shape"])
            tensor = np.array(tensor_data["data"], dtype=np.float32)
            tensor = tensor.reshape(shape)
            assert shape == (1, game.BOARD_SIZE, game.BOARD_SIZE, game.PLANES_NUM)

            # Run model from Python and assert all outputs are equal
            py_outputs = [model(tensor) for _ in range(ASSERT_PYTHON_OUTPUT_EQ_REPEAT)]
            py_outputs = [
                (val.numpy().item(), probs.numpy()) for (val, probs) in py_outputs
            ]
            py_output = py_outputs[0]
            for out_other in py_outputs:
                assert is_outputs_equals(py_output, out_other)

            # Run model from Rust and assert all outputs are equal
            subprocess.run(
                [
                    "cargo",
                    "run",
                    "--profile",
                    "dev",
                    "-q",
                    "--bin",
                    "test_net_output",
                    "--",
                    "--game",
                    game_name,
                    "--position",
                    position,
                    "--model-path",
                    model_path,
                    "--outfile",
                    output_file,
                    "--repeat",
                    str(ASSERT_RUST_OUTPUT_EQ_REPEAT),
                ],
                stderr=sys.stderr,
                stdout=sys.stdout,
                check=True,
                cwd=CATTUS_ENGINE_TOP,
            )
            with open(output_file, "r") as output_file:
                output = json.load(output_file)
            rs_outputs = zip(output["vals"], output["probs"])
            rs_outputs = [(val, np.array(probs)) for (val, probs) in rs_outputs]
            rs_output = rs_outputs[0]
            for out_other in rs_outputs:
                assert is_outputs_equals(rs_output, out_other)

            # Assert Python output and Rust output are equal
            assert is_outputs_equals(py_output, rs_output)


def create_model(game: TrainableGame, path: str) -> keras.Model:
    cfg = {
        "model": {
            "residual_filter_num": 1,
            "residual_block_num": 1,
            "value_head_conv_output_channels_num": 1,
            "policy_head_conv_output_channels_num": 1,
        },
        "cpu": True,
    }
    model = game.create_model("ConvNetV1", cfg)

    input_signature = game.model_input_signature("ConvNetV1", cfg)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(onnx_model, path)

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
        ],
    )


def test_hex_net_output():
    _test_net_output(
        "hex11",
        Hex(11),
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
            "r",
        ],
    )


def test_chess_net_output():
    _test_net_output(
        "chess",
        Chess(),
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "nnqrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNQRKBBR w - - 0 1",
            "8/1p6/3QR3/6k1/1P2b3/2P3K1/6b1/6r1/ w - - 0 1",
            "4k2r/6r1/8/8/8/8/3R4/R3K3 w Qk - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1",
        ],
    )


if __name__ == "__main__":
    test_ttt_net_output()
    test_hex_net_output()
    test_chess_net_output()
    logging.info("test passed")
