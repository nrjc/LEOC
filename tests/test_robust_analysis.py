import unittest

import numpy as np

from examples.envs_utils import myPendulum, myCartpole, myMountainCar
from pilco.controller_utils import LQR
from pilco.controllers import CombinedController, LinearController
from pilco.noise_robust_analysis import percentage_stable, analyze_robustness
from utils import load_controller_from_obj


class TestRobustNess(unittest.TestCase):
    def setUp(self):
        bf = 60
        max_action = 50.0
        target = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Set up objects and variables
        # env = myPendulum(True)
        # env = myMountainCar(True)
        env = myCartpole(True)
        A, B, C, Q = env.control()
        W_matrix = LQR().get_W_matrix(A, B, Q, env='cartpole')

        state_dim = 5
        control_dim = 1

        self.controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                                             controller_location=target, W=W_matrix, max_action=max_action)
        self.lin_controller = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix,
                                         max_action=max_action)
        self.rbf_controller = load_controller_from_obj('/Users/Naifu/Desktop/PILCO2/PILCO/examples/controllers/cartpole/cartpole_controller10.pkl')
        self.env = env

    def test_percentage_stable(self):
        percentage_stable(self.controller, self.env, [(0.5, 1.2), (-np.pi / 4, np.pi / 4), (-1, 1)], ['g', 'm', 'l'], 0.2)

    def test_stable_across_noise(self):
        # p_extended = analyze_robustness(self.controller, self.env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2), (-0.02, 0.02), (-1, 1)], ['masscart', 'masspole', 'length'],
        #                   np.asarray([0.7, 1.0]))
        # # pass
        # p_linear = analyze_robustness(self.lin_controller, self.env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2), (-0.02, 0.02), (-1, 1)], ['masscart', 'masspole', 'length'],
        #                   np.asarray([0.7, 1.0]))
        p_rbf = analyze_robustness(self.rbf_controller, self.env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2), (-0.0192, 0.0192), (-1, 1)], ['masscart', 'masspole', 'length'],
                          [0.5, 0.7, 1.0])
        pass

if __name__ == '__main__':
    unittest.main()
