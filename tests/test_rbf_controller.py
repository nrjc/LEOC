import unittest
import numpy as np
import tensorflow as tf
from utils import load_controller_from_obj
from pilco.controllers import RbfController, LinearController, squash_sin
import numpy as np
import os
import tensorflow as tf

from gpflow import config

float_type = config.default_float()


def test_linearize():
    from pilco.controllers import RbfController
    rbf_controller = RbfController(3, 1, 10)
    action = \
        rbf_controller.compute_action(tf.reshape(tf.convert_to_tensor([1., 0., 0.], dtype=tf.dtypes.float64), (1, -1)),
                                      tf.zeros([3, 3], dtype=tf.dtypes.float64),
                                      squash=True)[0]
    lin_info = rbf_controller.linearize(np.array([1., 0., 0.]))
    assert len(lin_info) == 1


def test_linearize_sample():
    from pilco.controllers import RbfController
    rbf_controller = RbfController(3, 1, 10)
    location_linear = np.array([1., 0., 0.])
    rbf_controller.compute_action(tf.reshape(tf.convert_to_tensor(location_linear, dtype=tf.dtypes.float64), (1, -1)),
                                  tf.zeros([3, 3], dtype=tf.dtypes.float64),
                                  squash=True)[0]
    lin_info = rbf_controller.linearize_sample(location_linear)
    assert len(lin_info) == 2


if __name__ == '__main__':
    test_linearize()
    test_linearize_sample()
