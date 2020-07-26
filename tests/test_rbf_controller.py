import unittest
import numpy as np

class TestControllers(unittest.TestCase):
    def test_linearize(self):
        from pilco.controllers import RbfController
        rbf_controller = RbfController(3, 1, 10)
        lin_info = rbf_controller.linearize(np.array([1., 0., 0.]))
        self.assertEqual(len(lin_info), 1)

if __name__ == '__main__':
    unittest.main()
