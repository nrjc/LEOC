import numpy as np

from pilco.envs_utils import myPendulum
from controller_utils import LQR
from pilco.controllers import CombinedController, LinearController
from pilco.noise_robust_analysis import percentage_stable, analyze_robustness

bf = 60
max_action = 50.0
target = np.array([0.0, 0.0, 0.0])

# Set up objects and variables
env = myPendulum(True)
# env = myMountainCar(True)
# env = myCartpole(True)
A, B, C, Q = env.control()
W_matrix = LQR().get_W_matrix(A, B, Q, env='swing up')

state_dim = 3
control_dim = 1

controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                                controller_location=target, W=W_matrix, max_action=max_action)
lin_controller = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix,
                                  max_action=max_action)

def test_percentage_stable():
    percentage_stable(controller, env, [(0.5, 1.2), (-np.pi / 4, np.pi / 4), (-1, 1)], ['g', 'm', 'l'], 0.2, 1)


def test_stable_across_noise():
    p_extended = analyze_robustness(controller, env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2)], ['m', 'l', 'g'], np.array([0.1]))
    # # pass
    # p_linear = analyze_robustness(self.lin_controller, self.env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2), (-0.02, 0.02), (-1, 1)], ['masscart', 'masspole', 'length'],
    #                   np.asarray([0.7, 1.0]))
    # p_rbf = analyze_robustness(controller, env, [(-0.1, 0.1), (-1, 1), (0.5, 1.2), (-0.0192, 0.0192), (-1, 1)],
    #                            ['m', 'l', 'g'],
    #                            [30])


if __name__ == '__main__':
    test_percentage_stable()
    test_stable_across_noise()
