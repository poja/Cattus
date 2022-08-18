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
    model.add(Dense(121, activation='relu', name="in_position",
              input_dim=121))
    model.add(Dense(1, activation='tanh', name="out_value"))

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def reverse_position(pos, winner):
    return [-x for x in pos], -winner


def train_model(model_path, data_dir, output_dir):
    model = tf.keras.models.load_model(model_path)
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
    print("Accuracy:", sum(acc) / len(acc))

    model.save(output_dir, save_format='tf')


if __name__ == "__main__":
    # If --create is provided, a new model will be created and saved to --out
    # If --train is provided, an existing model will be loaded from --model, trained on --data and saved to --out
    # If both --create and --train are provided, a new model will be created, trained on --data and saved to --out
    parser = argparse.ArgumentParser(description="Model creator")
    parser.add_argument("--create", action='store_true', help="Create a model")
    parser.add_argument("--out", dest="output_dir", required=True,
                        type=str, help="output directory for the model")
    parser.add_argument("--train", action='store_true', help="Trains a model")
    parser.add_argument("--data", dest="data_dir", type=str,
                        help="data directory to train on")
    parser.add_argument("--model", dest="model_path",
                        type=str, help="path to existing model")
    args = parser.parse_args()

    if args.create:
        create_model().save(args.output_dir, save_format='tf')
        model_path = args.output_dir
    else:
        model_path = args.model_path

    if args.train:
        train_model(model_path, args.data_dir, args.output_dir)
