#!/bin/usr/env python3

# Run this file to create a model to be used (training and predicting) in Rust

import argparse
import tensorflow as tf


# tf.device("/cpu:0")


def make_model(output_path):
    inputs = tf.keras.Input(shape=(112, 8, 8))

    flow = inputs
    # add more layers by the followings:
    # flow = ...(flow)
    # flow = tf.keras.layers.Dense(64)(flow)
    flow = tf.keras.layers.Dense(1)(flow)

    # can support multiple outputs in the future
    outputs = [flow]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr = 0.01  # changing LR in the future
    # lr_var = tf.Variable(0.01, trainable=False)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr,
        # learning_rate=lambda: self.active_lr,
        momentum=0.9, nesterov=True)

    def loss_func(target, output):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(target, output)
    loss = loss_func

    tf.train.Checkpoint(optimizer=optimizer, model=model).save(output_path)

    # # Get concrete function for the call method
    # pred_output = model.call.get_concrete_function(
    #     tf.TensorSpec(shape=[1, 2], dtype=tf.float32, name='inputs'))

    # # Get concrete function for the training method
    # train_output = model.training.get_concrete_function((tf.TensorSpec(
    #     shape=[1, 2], dtype=tf.float32, name="training_input"), tf.TensorSpec(shape=[1, 1], dtype=tf.float32, name="training_target")))

    # # Saving the model, explicitly adding the concrete functions as signatures
    # model.save(output_path, save_format='tf', signatures={
    #     'train': train_output, 'pred': pred_output})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model creator")
    parser.add_argument("--out", dest="output_path", type=str,
                        required=True, help="output directory")
    args = parser.parse_args()

    make_model(args.output_path)
