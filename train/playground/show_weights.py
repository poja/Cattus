import json
import tensorflow as tf
from train import net_utils
import numpy as np
from train.playground.ttt_representations import Ttt

model_path = '/Users/yishai/work/RL/workarea_nettest/models/myfit2'
custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print(weights)
    
