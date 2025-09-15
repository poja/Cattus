from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from construct import Array, Float32l, Int8sl, Int8ul, Int64ul, Struct
from torch import Tensor

from cattus_train import net_utils
from cattus_train.trainable_game import DataEntryParseError, Game


class NetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class Chess(Game):
    BOARD_SIZE = 8
    PLANES_NUM = 18
    MOVE_NUM = 1880

    ENTRY_FORMAT = Struct(
        "planes" / Array(PLANES_NUM, Int64ul),
        "moves_bitmap" / Array(235, Int8ul),
        "probs" / Array(225, Float32l),
        "winner" / Int8sl,
    )

    def load_data_entry(self, path: Path) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        with open(path, "rb") as f:
            entry_bytes = f.read()
        if len(entry_bytes) != self.ENTRY_FORMAT.sizeof():
            raise DataEntryParseError(
                "invalid training data file: {} ({} != {})".format(path, len(entry_bytes), self.ENTRY_FORMAT.sizeof())
            )
        entry = self.ENTRY_FORMAT.parse(entry_bytes)
        planes = torch.tensor(entry.planes, dtype=torch.uint64)
        moves_bitmap = np.array(entry.moves_bitmap, dtype=np.uint8)
        probs = np.array(entry.probs, dtype=np.float32)
        winner = torch.tensor(float(entry.winner), dtype=torch.float32)

        probs_all = torch.full((self.MOVE_NUM,), -1.0, dtype=torch.float32)
        # TODO: for sure there is a more Pythonic way to do this loop
        probs_idx = 0
        for move_idx in range(self.MOVE_NUM):
            i, j = move_idx // 8, move_idx % 8
            if moves_bitmap[i] & (1 << j) != 0:
                probs_all[move_idx] = float(probs[probs_idx])
                probs_idx += 1
        probs = probs_all

        assert len(planes) == self.PLANES_NUM
        assert len(probs) == self.MOVE_NUM
        return (planes, (probs, winner))

    def _get_input_shape(self):
        return (1, self.PLANES_NUM, self.BOARD_SIZE, self.BOARD_SIZE)

    def _create_model_convnetv1(self, cfg: dict):
        return net_utils.ConvNetV1(
            input_shape=self._get_input_shape(),
            residual_block_num=cfg["residual_block_num"],
            residual_filter_num=cfg["residual_filter_num"],
            value_head_conv_output_channels_num=cfg["value_head_conv_output_channels_num"],
            policy_head_conv_output_channels_num=cfg["policy_head_conv_output_channels_num"],
            moves_num=self.MOVE_NUM,
        )

    def create_model(self, net_type: str, cfg: dict) -> nn.Module:
        if net_type == NetType.SimpleTwoHeaded:
            return net_utils.SimpleTwoHeadedModel(self._get_input_shape(), self.MOVE_NUM)
        elif net_type == NetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def model_input_shape(self, net_type: str) -> tuple:
        if net_type == NetType.SimpleTwoHeaded or net_type == NetType.ConvNetV1:
            return self._get_input_shape()
        else:
            raise ValueError("Unknown model type: " + net_type)
