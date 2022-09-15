#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import struct
import random
import functools
from tictactoe import TicTacToe


class DataParser:
    def __init__(self, game, data_dir):
        self.game = game
        self.data_dir = data_dir

    def _data_entries_filenames_gen(self):
        filenames = os.listdir(self.data_dir)
        filenames = [os.path.join(self.data_dir, filename)
                     for filename in filenames]

        # take the latests files_count files
        files_count = 100000
        filenames.sort(key=os.path.getmtime, reverse=True)
        if len(filenames) > files_count:
            filenames = filenames[:files_count]

        random.shuffle(filenames)

        for filename in filenames:
            yield os.path.join(self.data_dir, filename)

    def _read_data_entry_gen(self, filenames_gen):
        for filename in filenames_gen:
            yield self.game.load_data_entry(filename)

    def _unpack_planes_gen(self, nparr_packed_gen):
        for (planes, probs, winner) in nparr_packed_gen:
            assert planes.dtype == np.uint32
            plane_size = self.game.BOARD_SIZE * self.game.BOARD_SIZE
            planes = [np.frombuffer(plane, dtype=np.uint8) for plane in planes]
            planes = [np.unpackbits(plane, count=plane_size)
                      for plane in planes]
            planes = np.array(planes, dtype=np.float32)
            yield (planes, probs, winner)

    def _serialize_gen(self, nparr_gen):
        for (planes, probs, winner) in nparr_gen:
            planes = planes.tobytes()
            plane_size = self.game.BOARD_SIZE * self.game.BOARD_SIZE
            assert len(planes) == (self.game.PLANES_NUM * plane_size * 4)

            assert len(probs) == self.game.MOVE_NUM
            probs = probs.astype('f').tostring()

            winner = struct.pack('f', winner)

            yield (planes, probs, winner)

    def generator(self):
        # choose entries
        filenames_gen = self._data_entries_filenames_gen()
        # filename -> tuple of np array with packed planes
        nparr_packed_gen = self._read_data_entry_gen(filenames_gen)
        # planes bitmap -> full planes arrays
        nparr_gen = self._unpack_planes_gen(nparr_packed_gen)
        # np tuple -> tuple of bytes
        bytes_gen = self._serialize_gen(nparr_gen)
        for x in bytes_gen:
            yield x

    def _parse_func(self, planes, probs, winner):
        """
        Convert unpacked record to tensors for tensorflow training
        """
        planes = tf.io.decode_raw(planes, tf.float32)
        probs = tf.io.decode_raw(probs, tf.float32)
        winner = tf.io.decode_raw(winner, tf.float32)

        # TODO shape of tensor might need to be [size][size][planes] instead
        planes = tf.reshape(planes, (-1, self.game.PLANES_NUM,
                            self.game.BOARD_SIZE, self.game.BOARD_SIZE))
        probs = tf.reshape(probs, (-1, self.game.MOVE_NUM))
        winner = tf.reshape(winner, (-1, 1))

        return (planes, probs, winner)

    def get_parse_func(self):
        return functools.partial(DataParser._parse_func, self)
