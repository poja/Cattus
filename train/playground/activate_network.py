import tensorflow as tf
from train import net_utils
import numpy as np

model_path = '/Users/yishai/work/RL/workarea_nettest/models/myfit'
custom_objects = {
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
output = model(np.array([0] * 27).reshape([1, 3, 3, 3]))
print(output)
# import IPython; IPython.embed()
