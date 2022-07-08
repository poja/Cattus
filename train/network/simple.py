#!/usr/bin/env python3

import argparse
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from keras import optimizers
import numpy as np
import json
import os
import random


def create_model():
    model = Sequential()
    model.add(Dense(121, activation='relu', name="test_in",
              input_dim=121))
    model.add(Dense(1, activation='tanh', name="test_out"))

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def reverse_position(pos, winner):
    return [-x for x in pos], -winner


def train_model(model, data_dir):
    positions = []
    winners = []

    for data_file in os.listdir(data_dir):
        with open(os.path.join(data_dir, data_file), "rb") as f:
            data_obj = json.load(f)
        pos, win = data_obj["position"], data_obj["winner"]
        if random.choice([True, False]):
            pos, win = reverse_position(pos, win)
        positions.append(pos)
        winners.append(win)
    model.fit(positions, winners, batch_size=4, epochs=16)

    preds = [x[0] for x in model.predict(positions)]
    preds = [1 if x >= 0 else -1 for x in preds]
    wins = [1 if x >= 0 else -1 for x in winners]
    acc = [preds[i] == wins[i] for i in range(len(preds))]
    print(sum(acc) / len(acc))


def make_train_and_save_model(output_dir, data_dir):
    model = create_model()
    train_model(model, data_dir)
    model.save(output_dir, save_format='tf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model creator")
    parser.add_argument("--out", dest="output_dir", type=str,
                        required=True, help="output directory")
    parser.add_argument("--data", dest="data_dir", type=str,
                        required=True, help="data directory")
    args = parser.parse_args()

    make_train_and_save_model(args.output_dir, args.data_dir)
