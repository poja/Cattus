#!/usr/bin/env python3

import sys
import json
import logging
import os
import shutil
import subprocess
import numpy as np

from train.hex import Hex
from train.tictactoe import TicTacToe
from train.chess import Chess
from train.data_parser import DataParser


REMOVE_TMP_DIR_ON_FINISH = True

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TMP_DIR = os.path.join(TESTS_DIR, "tmp", "test_serialize_encode")
SERIALIZE_FILE = os.path.join(TMP_DIR, "serialize_res.json")
ENCODE_FILE = os.path.join(TMP_DIR, "encode_res.json")

logging.basicConfig(
    level=logging.DEBUG,
    format="[Serialize Encode Test]: %(message)s")


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
        ]
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
            "r"
        ]
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
        ]
    )


def _test_serialize_encode(game_name, game, positions):
    for position in positions:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_DIR)

        try:
            subprocess.run([
                "cargo", "run", "--profile", "dev", "-q", "--bin",
                "test_encode", "--",
                "--game", game_name,
                "--position", position,
                "--outfile", ENCODE_FILE],
                stderr=sys.stderr, stdout=sys.stdout, check=True)
            subprocess.run([
                "cargo", "run", "--profile", "dev", "-q", "--bin",
                "test_serialize", "--",
                "--game", game_name,
                "--position", position,
                "--outfile", SERIALIZE_FILE],
                stderr=sys.stderr, stdout=sys.stdout, check=True)

            cpu = True
            packed_entry = game.load_data_entry(SERIALIZE_FILE)
            nparr_entry = DataParser.unpack_planes(packed_entry, game, cpu)
            bytes_entry = DataParser.serialize(nparr_entry, game)
            data_parser_tensor = DataParser.bytes_entry_to_tensor(
                bytes_entry, game, cpu)
            planes_dpt, _ = data_parser_tensor

            with open(ENCODE_FILE, "r") as encode_file:
                rust_tensor_data = json.load(encode_file)

            planes_rust_shape = tuple(rust_tensor_data["shape"])
            if planes_rust_shape != planes_dpt.shape:
                print("game", game_name)
                print("position", position)
                raise ValueError("planes tensor shape mismatch",
                                 planes_rust_shape, planes_dpt.shape)

            planes_dpt = planes_dpt.numpy().flatten()
            planes_rust = np.array(
                rust_tensor_data["data"], dtype=np.float32)
            if (planes_rust != planes_dpt).any():

                raise ValueError(f"Planes tensor mismatch."
                                 f"(Game: {game_name}, Position: {position}, "
                                 f"Rust planes: {planes_rust}, Data parser planes: {planes_dpt})")
        finally:
            if REMOVE_TMP_DIR_ON_FINISH:
                shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    test_ttt_serialize_encode()
    test_hex_serialize_encode()
    test_chess_serialize_encode()
    logging.info("test passed")
