#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import struct
import random
import functools


class DataParser:
    def __init__(self, game, data_dir, entries_count):
        self.game = game
        self.data_dir = data_dir
        self.entries_count = entries_count
        self.cpu = True

    def _data_entries_filenames_gen(self):
        filenames = os.listdir(self.data_dir)
        filenames = [os.path.join(self.data_dir, filename)
                     for filename in filenames]

        # take the latests files
        filenames.sort(key=os.path.getmtime, reverse=True)
        if len(filenames) > self.entries_count:
            filenames = filenames[:self.entries_count]

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
            planes = np.reshape(
                planes, (self.game.PLANES_NUM, self.game.BOARD_SIZE, self.game.BOARD_SIZE))
            if self.cpu:
                planes = np.transpose(planes, (1, 2, 0))
            yield (planes, probs, winner)

    def _serialize_gen(self, nparr_gen):
        f32size = np.dtype(np.float32).itemsize
        for (planes, probs, winner) in nparr_gen:
            planes = planes.tobytes()
            plane_size = self.game.BOARD_SIZE * self.game.BOARD_SIZE
            assert len(planes) == (self.game.PLANES_NUM * plane_size * f32size)

            probs = probs.astype('f').tostring()
            assert len(probs) == self.game.MOVE_NUM * f32size

            winner = struct.pack('f', winner)
            assert len(winner) == 1 * f32size

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

        yield from bytes_gen

    def _parse_func(self, planes, probs, winner):
        """
        Convert unpacked record to tensors for tensorflow training
        """
        planes = tf.io.decode_raw(planes, tf.float32)
        probs = tf.io.decode_raw(probs, tf.float32)
        winner = tf.io.decode_raw(winner, tf.float32)

        planes_shape_cpu = (-1, self.game.BOARD_SIZE,
                            self.game.BOARD_SIZE, self.game.PLANES_NUM)
        planes_shape_gpu = (-1, self.game.PLANES_NUM,
                            self.game.BOARD_SIZE, self.game.BOARD_SIZE)
        planes_shape = planes_shape_cpu if self.cpu else planes_shape_gpu
        planes = tf.reshape(planes, planes_shape)
        probs = tf.reshape(probs, (-1, self.game.MOVE_NUM))
        winner = tf.reshape(winner, (-1, 1))

        return (planes, probs, winner)

    def get_parse_func(self):
        return functools.partial(DataParser._parse_func, self)
