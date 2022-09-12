#!/usr/bin/env python3

from abc import ABC

import keras


class NetCategory:
    Scalar = "scalar"
    TwoHeaded = "two_headed"


class TrainableGame(ABC):

    def create_model(self, net_type: str) -> keras.Model:
        pass

    def load_model(self, path: str, net_type: str) -> keras.Model:
        pass

    def get_net_category(self, net_type: str) -> NetCategory:
        pass

    def load_data_entry(self, path: str) -> dict:
        pass
