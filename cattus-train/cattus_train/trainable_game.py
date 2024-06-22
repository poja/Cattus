from abc import ABC, abstractmethod

import keras
import tensorflow as tf


class TrainableGame(ABC):
    @abstractmethod
    def create_model(self, net_type: str, cfg: dict) -> keras.Model:
        ...

    @abstractmethod
    def model_input_signature(self, net_type: str, cfg: dict) -> list[tf.TensorSpec]:
        ...

    @abstractmethod
    def load_model(self, path: str, net_type: str) -> keras.Model:
        ...

    @abstractmethod
    def load_data_entry(self, path: str, cfg: dict) -> ():
        ...


class DataEntryParseError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
