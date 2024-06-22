import tensorflow as tf
from cattus_train.net_utils import (
    loss_cross_entropy,
    policy_head_accuracy,
    value_head_accuracy,
)


class Model:
    def __init__(self, model_path: str):
        self._model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "loss_cross_entropy": loss_cross_entropy,
                "value_head_accuracy": value_head_accuracy,
                "policy_head_accuracy": policy_head_accuracy,
            },
        )

    def call(self, x_shape: list[int], x_data: list[float]):
        x = tf.reshape(tf.convert_to_tensor(x_data), x_shape)
        n = x.shape[0]

        outputs = self._model(x)
        assert len(outputs) == 2
        vals, probs = outputs[0], outputs[1]

        assert vals.shape == (n, 1)
        assert probs.shape[0] == n
        vals = vals.numpy().reshape(n).tolist()
        probs = probs.numpy().reshape(n, -1).tolist()
        return vals, probs
