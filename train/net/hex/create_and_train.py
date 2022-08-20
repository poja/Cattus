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
import json
import os
import hex_utils

BATCH_SIZE = 4
EPOCHS = 16

def create_model():
    input_layer = Input(shape=(121), name="in_position")
    x = Dense(units="121", activation="relu")(input_layer)
    output_layer = Dense(units="1", activation="tanh", name="out_value")(x)

    model = Model(inputs=input_layer, outputs=[output_layer])

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def train_model(model_path, data_dir, output_dir):
    model = tf.keras.models.load_model(model_path)
    positions, winners = [], []

    for data_file in os.listdir(data_dir):
        data_entry = hex_utils.load_data_entry(os.path.join(data_dir, data_file))
        positions.append(data_entry["position"])
        winners.append(data_entry["winner"])

    model.fit(positions, winners, batch_size=BATCH_SIZE, epochs=EPOCHS)

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
