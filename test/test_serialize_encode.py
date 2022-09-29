#!/usr/bin/env python3

import sys
import json
import logging
import os
import shutil
import subprocess
import numpy as np


DEBUG = True
REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
RL_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
TRAIN_TOP = os.path.join(RL_TOP, "train")
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_serialize_encode")
SERIALIZE_FILE = os.path.join(TMP_DIR, "serialize_res.json")
ENCODE_FILE = os.path.join(TMP_DIR, "encode_res.json")

if TRAIN_TOP not in sys.path:
    sys.path.insert(0, TRAIN_TOP)

from hex import Hex
from tictactoe import TicTacToe
from chess import Chess
from data_parser import DataParser


def run_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[Serialize Encode Test]: %(message)s')

    games_args = [
        {
            "game_name": "tictactoe",
            "game_obj": TicTacToe(),
            "positions": [
                "___x__o_ox",
                "oox__oo_xo",
                "o_xx_x__ox",
                "o__xo__xxx",
                "o___x_o__x",
                "_o___xx__o",
                "oo__x____x",
                "oo__o__oxx",
            ]
        },
        {
            "game_name": "hex",
            "game_obj": Hex(),
            "positions": [
                "reeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
eeeeeeeeeee\
r",
                "rererererer\
ererererere\
rererererer\
ererererere\
rererererer\
ererererere\
rererererer\
ererererere\
rererererer\
ererererere\
rererererer\
r",
                "reeeeeeeeee\
ereeeeeeeee\
eereeeeeeee\
eeereeeeeee\
eeeereeeeee\
eeeeereeeee\
eeeeeereeee\
eeeeeeereee\
eeeeeeeeree\
eeeeeeeeere\
eeeeeeeeeer\
r"
            ]
        },
        {
            "game_name": "chess",
            "game_obj": Chess(),
            "positions": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "nnqrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNQRKBBR w - - 0 1",
                "8/1p6/3QR3/6k1/1P2b3/2P3K1/6b1/6r1/ w - - 0 1",
                "4k2r/6r1/8/8/8/8/3R4/R3K3 w Qk - 0 1"
                # "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" TODO
            ]
        }
    ]

    for args in games_args:
        for position in args["positions"]:
            if os.path.exists(TMP_DIR):
                shutil.rmtree(TMP_DIR)
            os.makedirs(TMP_DIR)

            try:
                subprocess.run([
                    "cargo", "run", "--profile", "dev", "-q", "--bin",
                    "test_serialize_encode", "--",
                    "--game", args["game_name"],
                    "--position", position,
                    "--serialize-out", SERIALIZE_FILE,
                    "--encode-out", ENCODE_FILE],
                    stderr=sys.stderr, stdout=sys.stdout, check=True)

                cpu = True
                game = args["game_obj"]
                planes_shape_cpu = (1, game.BOARD_SIZE,
                                    game.BOARD_SIZE, game.PLANES_NUM)
                planes_shape_gpu = (1, game.PLANES_NUM,
                                    game.BOARD_SIZE, game.BOARD_SIZE)
                planes_shape = planes_shape_cpu if cpu else planes_shape_gpu
                packed_entry = game.load_data_entry(SERIALIZE_FILE)
                nparr_entry = DataParser.unpack_planes(packed_entry, game, cpu)
                bytes_entry = DataParser.serialize(nparr_entry, game)
                data_parser_tensor = DataParser.bytes_entry_to_tensor(
                    bytes_entry, game, cpu)
                planes_dpt, _probs_dpt, _winner_dpt = data_parser_tensor

                with open(ENCODE_FILE, "r") as encode_file:
                    rust_tensor_data = json.load(encode_file)

                planes_rust_shape = tuple(rust_tensor_data["shape"])
                if planes_rust_shape != planes_dpt.shape:
                    print("game", args["game_name"])
                    print("position", position)
                    raise ValueError("planes tensor shape mismatch",
                                     planes_rust_shape, planes_dpt.shape)

                planes_dpt = planes_dpt.numpy().flatten()
                planes_rust = np.array(
                    rust_tensor_data["data"], dtype=np.float32)
                if (planes_rust != planes_dpt).any():
                    print("game", args["game_name"])
                    print("position", position)
                    # print("rust planes", planes_rust.reshape(planes_shape))
                    # print("data parser planes", planes_dpt.reshape(planes_shape))
                    print("rust planes", planes_rust)
                    print("data parser planes", planes_dpt)
                    raise ValueError("planes tensor mismatch")
            finally:
                if REMOVE_TMP_DIR_ON_FINISH:
                    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    run_test()
