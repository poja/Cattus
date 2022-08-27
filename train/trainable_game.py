from abc import ABC

import keras


class NetType:
    SimpleScalar = "simple_scalar"
    SimpleTwoHeaded = "simple_two_headed"


class TrainableGame(ABC):

    def create_model_simple_scalar(self) -> keras.Model:
        pass

    def create_model_simple_two_headed(self) -> keras.Model:
        pass

    def load_data_entry(self, path: str) -> dict:
        pass
