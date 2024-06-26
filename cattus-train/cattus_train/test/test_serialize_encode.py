import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np

from cattus_train.chess import Chess
from cattus_train.data_parser import DataParser
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_ENGINE_TOP = os.path.abspath(
    os.path.join(TESTS_DIR, "..", "..", "..", "cattus-engine")
)

logging.basicConfig(level=logging.DEBUG, format="[Serialize Encode Test]: %(message)s")


def test_ttt_serialize_encode():
    _test_serialize_encode(
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


def test_hex_serialize_encode():
    _test_serialize_encode(
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


def test_chess_serialize_encode():
    _test_serialize_encode(
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


def _test_serialize_encode(game_name, game, positions):
    for position in positions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            serialize_file = os.path.join(tmp_dir, "serialize_res.json")
            encode_file = os.path.join(tmp_dir, "encode_res.json")

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
                    encode_file,
                ],
                stderr=sys.stderr,
                stdout=sys.stdout,
                check=True,
                cwd=CATTUS_ENGINE_TOP,
            )
            subprocess.run(
                [
                    "cargo",
                    "run",
                    "--profile",
                    "dev",
                    "-q",
                    "--bin",
                    "test_serialize",
                    "--",
                    "--game",
                    game_name,
                    "--position",
                    position,
                    "--outfile",
                    serialize_file,
                ],
                stderr=sys.stderr,
                stdout=sys.stdout,
                check=True,
                cwd=CATTUS_ENGINE_TOP,
            )

            cpu = True
            packed_entry = game.load_data_entry(serialize_file)
            nparr_entry = DataParser.unpack_planes(packed_entry, game, cpu)
            bytes_entry = DataParser.serialize(nparr_entry, game)
            data_parser_tensor = DataParser.bytes_entry_to_tensor(
                bytes_entry, game, cpu
            )
            planes_dpt, _ = data_parser_tensor

            with open(encode_file, "r") as encode_file:
                rust_tensor_data = json.load(encode_file)

            planes_rust_shape = tuple(rust_tensor_data["shape"])
            if planes_rust_shape != planes_dpt.shape:
                print("game", game_name)
                print("position", position)
                raise ValueError(
                    "planes tensor shape mismatch", planes_rust_shape, planes_dpt.shape
                )

            planes_dpt = planes_dpt.numpy().flatten()
            planes_rust = np.array(rust_tensor_data["data"], dtype=np.float32)
            if (planes_rust != planes_dpt).any():
                raise ValueError(
                    f"Planes tensor mismatch."
                    f"(Game: {game_name}, Position: {position}, "
                    f"Rust planes: {planes_rust}, Data parser planes: {planes_dpt})"
                )


if __name__ == "__main__":
    test_ttt_serialize_encode()
    test_hex_serialize_encode()
    test_chess_serialize_encode()
    logging.info("test passed")
