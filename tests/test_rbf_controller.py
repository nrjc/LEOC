import unittest
import numpy as np
import tensorflow as tf
from utils import load_controller_from_obj
class TestControllers(unittest.TestCase):
    def test_linearize(self):
        from pilco.controllers import RbfController
        rbf_controller = load_controller_from_obj('data/swingup/rbf/swingup_rbf_controller4.pkl')
        action = rbf_controller.compute_action(tf.reshape(tf.convert_to_tensor([1.,0.,0.], dtype=tf.dtypes.float64), (1, -1)),
                                                  tf.zeros([3, 3], dtype=tf.dtypes.float64),
                                                  squash=True)[0]
        lin_info = rbf_controller.linearize(np.array([1., 0., 0.]))
        self.assertEqual(len(lin_info), 1)
    def test_linearize_sample(self):
        from pilco.controllers import RbfController
        rbf_controller = load_controller_from_obj('data/swingup/rbf/swingup_rbf_controller4.pkl')
        location_linear = np.array([1.,0.,0.])
        rbf_controller.compute_action(tf.reshape(tf.convert_to_tensor(location_linear, dtype=tf.dtypes.float64), (1, -1)),
                                      tf.zeros([3, 3], dtype=tf.dtypes.float64),
                                      squash=True)[0]
        lin_info = rbf_controller.linearize_sample(location_linear)

        self.assertEqual(len(lin_info), 1)
if __name__ == '__main__':
    unittest.main()
