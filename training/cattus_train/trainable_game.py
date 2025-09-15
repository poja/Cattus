from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from torch import Tensor


class Game(ABC):
    @abstractmethod
    def create_model(self, net_type: str, cfg: dict) -> nn.Module:
        ...

    @abstractmethod
    def model_input_shape(self, net_type: str) -> tuple:
        ...

    @abstractmethod
    def load_data_entry(self, path: Path) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        ...


class DataEntryParseError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
