import os
import random
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from cattus_train.trainable_game import DataEntryParseError, Game


class DataSet(IterableDataset):
    def __init__(self, game: Game, train_data_dir: Path, cfg: dict, device: torch.device):
        self._game: Game = game
        self._train_data_dir: Path = train_data_dir
        self._cfg: dict = cfg
        self._device: torch.device = device

        assert self._cfg["training"]["latest_data_entries"] >= self._cfg["training"]["iteration_data_entries"]

    def _data_entries_filenames_gen(self) -> Generator[int, None, None]:
        filenames = [p for p in Path(self._train_data_dir).rglob("*.traindata")]

        # take the latests files
        latest_de = self._cfg["training"]["latest_data_entries"]
        filenames.sort(key=os.path.getmtime, reverse=True)
        if len(filenames) > latest_de:
            filenames = filenames[:latest_de]

        # take a sub set
        iter_de = self._cfg["training"]["iteration_data_entries"]
        random.shuffle(filenames)
        if len(filenames) > iter_de:
            filenames = filenames[:iter_de]

        for filename in filenames:
            yield self._train_data_dir / filename

    def _read_data_entry_gen(self, filenames_gen) -> Generator[tuple[Tensor, tuple[Tensor, Tensor]], None, None]:
        for filename in filenames_gen:
            try:
                yield self._game.load_data_entry(filename)
            except DataEntryParseError:
                pass

    @staticmethod
    def unpack_planes(
        packed_entry: tuple[Tensor, tuple[Tensor, Tensor]], game: Game
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        planes, (probs, winner) = packed_entry
        assert len(planes) == game.PLANES_NUM
        plane_size = game.BOARD_SIZE * game.BOARD_SIZE
        planes = [np.frombuffer(plane.numpy(), dtype=np.uint8) for plane in planes]
        planes = np.array([np.unpackbits(plane, count=plane_size, bitorder="little") for plane in planes])
        planes = torch.tensor(planes, dtype=torch.float32)
        planes = planes.reshape((game.PLANES_NUM, game.BOARD_SIZE, game.BOARD_SIZE))
        return planes, (probs, winner)

    def _unpack_planes_gen(self, nparr_packed_gen) -> Generator[tuple[Tensor, tuple[Tensor, Tensor]], None, None]:
        for packed_entry in nparr_packed_gen:
            yield DataSet.unpack_planes(packed_entry, self._game)

    def _move_to_device(self, tensors_gen) -> Generator[tuple[Tensor, tuple[Tensor, Tensor]], None, None]:
        for planes, (probs, winner) in tensors_gen:
            yield planes.to(self._device), (probs.to(self._device), winner.to(self._device))

    def __iter__(self) -> Generator[tuple[Tensor, tuple[Tensor, Tensor]], None, None]:
        # choose entries
        filenames_gen = self._data_entries_filenames_gen()
        # filename -> tuple of np array with packed planes
        packed_tensors_gen = self._read_data_entry_gen(filenames_gen)
        # planes bitmap -> full planes arrays
        tensors_gen = self._unpack_planes_gen(packed_tensors_gen)
        # move to device
        tensors_on_device_gen = self._move_to_device(tensors_gen)

        yield from tensors_on_device_gen
