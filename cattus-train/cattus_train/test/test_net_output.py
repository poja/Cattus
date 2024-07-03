import json
import logging
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

from cattus_train.chess import Chess
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import Game

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_ENGINE_TOP = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", "..", "cattus-engine"))

ASSERT_PYTHON_OUTPUT_EQ_REPEAT = 8
ASSERT_RUST_OUTPUT_EQ_REPEAT = 8


logging.basicConfig(level=logging.DEBUG, format="[Net Output Test]: %(message)s")


def is_outputs_equals(o1, o2):
    probs1, val1 = o1
    probs2, val2 = o2
    return (
        math.isclose(val1, val2, rel_tol=1e-5, abs_tol=1e-6) and np.isclose(probs1, probs2, rtol=1e-3, atol=1e-6).all()
    )


def _test_net_output(game_name: str, game: Game, positions):
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
            tensor = torch.tensor(tensor_data["data"], dtype=torch.float32).reshape(tuple(tensor_data["shape"]))
            tensor = tensor[None, ...]  # Add batch dimension
            assert tensor.shape == (
                1,
                game.PLANES_NUM,
                game.BOARD_SIZE,
                game.BOARD_SIZE,
            )

            # Run model from Python and assert all outputs are equal
            py_outputs = [model(tensor) for _ in range(ASSERT_PYTHON_OUTPUT_EQ_REPEAT)]
            py_outputs = [(probs.detach().numpy(), val.detach().numpy().item()) for (probs, val) in py_outputs]
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
            rs_outputs = [(np.array(probs), val) for (probs, val) in zip(output["probs"], output["vals"])]
            rs_output = rs_outputs[0]
            for out_other in rs_outputs:
                assert is_outputs_equals(rs_output, out_other)

            # Assert Python output and Rust output are equal
            assert is_outputs_equals(py_output, rs_output)


def create_model(game: Game, path: str) -> nn.Module:
    cfg = {
        "model": {
            "residual_filter_num": 1,
            "residual_block_num": 1,
            "value_head_conv_output_channels_num": 1,
            "policy_head_conv_output_channels_num": 1,
        },
    }
    model = game.create_model("ConvNetV1", cfg)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(game.model_input_shape("ConvNetV1")),
            path,
            verbose=False,
            input_names=["planes"],
            output_names=["policy", "value"],
            dynamic_axes={"planes": {0: "batch"}},  # TODO: consider removing this, may affect performance
        )

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
