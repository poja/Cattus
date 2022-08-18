#!/usr/bin/env python3

import os
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
import hex_utils
import json
import argparse


BATCH_SIZE = 4
EPOCHS = 16
LEARNING_RATE = 0.001


class NetType:
    SimpleScalar = "simple_scalar"
    SimpleTwoHeaded = "simple_two_headed"


def create_model_simple_scalar():
    input_layer = Input(shape=(121), name="in_position")
    x = Dense(units="121", activation="relu")(input_layer)
    output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

    model = Model(inputs=input_layer, outputs=[output_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def create_model_simple_two_headed():
    input_layer = Input(shape=(121), name="in_position")
    x = Dense(units="121", activation="relu")(input_layer)
    output_scalar_layer = Dense(
        units="1", activation="tanh", name="out_value")(x)
    output_probs_layer = Dense(
        units="121", activation="sigmoid", name="out_probs")(x)

    model = Model(inputs=input_layer, outputs=[
                  output_scalar_layer, output_probs_layer])

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=opt,
        loss={'out_value': 'mse', 'out_probs': 'kl_divergence'},
        metrics={'out_value': tf.keras.metrics.RootMeanSquaredError(),
                 'out_probs': tf.keras.metrics.KLDivergence()})
    return model


def create_model(nettype):
    if nettype == NetType.SimpleScalar:
        return create_model_simple_scalar()
    if nettype == NetType.SimpleTwoHeaded:
        return create_model_simple_two_headed()
    raise ValueError("Unknown model type: " + nettype)


def train_model(model_path, data_dir, out_dir, nettype):
    model = tf.keras.models.load_model(model_path)
    xs, ys = [], []

    for data_file in os.listdir(data_dir):
        data_filename = os.path.join(data_dir, data_file)
        data_entry = hex_utils.load_data_entry(data_filename)
        xs.append(data_entry["position"])
        if nettype == NetType.SimpleScalar:
            ys.append(data_entry["winner"])
        elif nettype == NetType.SimpleTwoHeaded:
            y = (data_entry["winner"], data_entry["moves_probabilities"])
            ys.append(y)
        else:
            raise ValueError("Unknown model type: " + nettype)

    xs = np.array(xs)
    if nettype == NetType.SimpleScalar:
        ys = np.array(ys)
    elif nettype == NetType.SimpleTwoHeaded:
        ys_raw = ys
        ys = {'out_value': np.array([y[0] for y in ys_raw]),
              'out_probs': np.array([y[1] for y in ys_raw])}

    model.fit(x=xs, y=ys, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # TODO remove this
    """
    preds = [x[0] for x in model.predict(xs)]
    preds = [1 if x >= 0 else -1 for x in preds]
    wins = [1 if x >= 0 else -1 for x in ys]
    acc = [preds[i] == wins[i] for i in range(len(preds))]
    print("Accuracy:", sum(acc) / len(acc))
    """

    model.save(out_dir, save_format='tf')


if __name__ == "__main__":
    # If --create is provided, a new model will be created and saved to --out-dir
    # If --train is provided, an existing model will be loaded from --model-path, trained on --data-dir and saved to --out-dir
    # If both --create and --train are provided, a new model will be created, trained on --data-dir and saved to --out-dir
    parser = argparse.ArgumentParser(description="Model creator")
    parser.add_argument(
        "--type", choices=[NetType.SimpleScalar, NetType.SimpleTwoHeaded], required=True, help="network type")
    parser.add_argument(
        "--create", action='store_true', help="Create a model")
    parser.add_argument(
        "--out-dir", required=True, type=str, help="output directory for the created model")
    parser.add_argument(
        "--train", action='store_true', help="Trains a model")
    parser.add_argument(
        "--data-dir", type=str, help="data directory to train on")
    parser.add_argument(
        "--model-path", type=str, help="path to existing model")
    args = parser.parse_args()

    if args.create:
        create_model(args.type).save(args.out_dir, save_format='tf')
        model_path = args.out_dir
    else:
        model_path = args.model_path

    if args.train:
        train_model(model_path, args.data_dir, args.out_dir, args.type)
