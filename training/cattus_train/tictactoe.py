from pathlib import Path

import torch
import torch.nn as nn
from construct import Array, Float32l, Int8sl, Int64ul, Struct
from torch import Tensor

from cattus_train import net_utils
from cattus_train.trainable_game import DataEntryParseError, Game


class TtoNetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class TicTacToe(Game):
    BOARD_SIZE = 3
    PLANES_NUM = 3
    MOVE_NUM = BOARD_SIZE * BOARD_SIZE

    ENTRY_FORMAT = Struct(
        "planes" / Array(PLANES_NUM, Int64ul),
        "probs" / Array(MOVE_NUM, Float32l),
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
        probs = torch.tensor(entry.probs, dtype=torch.float32)
        winner = torch.tensor(float(entry.winner), dtype=torch.float32)

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
        if net_type == TtoNetType.SimpleTwoHeaded:
            return net_utils.SimpleTwoHeadedModel(self._get_input_shape(), self.MOVE_NUM)
        elif net_type == TtoNetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def model_input_shape(self, net_type: str) -> tuple:
        if net_type == TtoNetType.SimpleTwoHeaded or net_type == TtoNetType.ConvNetV1:
            return self._get_input_shape()
        else:
            raise ValueError("Unknown model type: " + net_type)
