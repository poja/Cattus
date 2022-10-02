import IPython
import json
import tensorflow as tf
from train import net_utils
import numpy as np
from train.playground.ttt_representations import Ttt

model_path = '/Users/yishai/work/RL/workarea_nettest/models/myfit2'
custom_objects = {
    "loss_const_0": net_utils.loss_const_0,
    "loss_cross_entropy": net_utils.loss_cross_entropy,
    "policy_head_accuracy": net_utils.policy_head_accuracy,
    "value_head_accuracy": net_utils.value_head_accuracy}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def check(s):
    output = model(Ttt.from_str(s).to_planes(True))
    print(output)


check('.........')
IPython.embed()
