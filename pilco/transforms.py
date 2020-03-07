from gpflow.transforms import Transform
import numpy as np
import tensorflow as tf
from gpflow._settings import SETTINGS as settings


class Squeeze(Transform):

    def forward(self, x):
        return x.squeeze(axis=0)

    def backward(self, y):
        return np.expand_dims(y, axis=0)

    def forward_tensor(self, x):
        return tf.squeeze(x, axis=0)

    def backward_tensor(self, y):
        return tf.expand_dims(y, axis=0)

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return "Squeeze"
