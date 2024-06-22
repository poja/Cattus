from abc import ABC

import keras


class TrainableGame(ABC):
    def create_model(self, net_type: str, cfg: dict) -> keras.Model:
        pass

    def load_model(self, path: str, net_type: str) -> keras.Model:
        pass

    def load_data_entry(self, path: str, cfg: dict) -> ():
        pass


class DataEntryParseError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
