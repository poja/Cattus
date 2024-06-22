import numpy as np
from onnxruntime import InferenceSession


class Model:
    def __init__(self, model_path: str):
        self._model = InferenceSession(model_path)

    def call(self, x_shape: list[int], x_data: list[float]):
        x = np.array(x_data, dtype=np.float32).reshape(x_shape)
        n = x.shape[0]

        outputs = self._model.run(None, {"input_planes": x})
        assert len(outputs) == 2
        vals, probs = outputs[0], outputs[1]

        assert vals.shape == (n, 1)
        assert probs.shape[0] == n
        vals = vals.reshape(n).tolist()
        probs = probs.reshape(n, -1).tolist()
        return vals, probs
