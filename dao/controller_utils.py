import numpy as np
import control
from gym import logger

class LQR:
    def get_W_matrix(self, A, B, Q, env='swingup'):
        # Compute gain matrix by solving Algebraic Riccati Equation
        # Reference https://www.mathworks.com/help/control/ref/lqr.html
        R = 1
        K, _, _ = control.lqr(A, B, Q, R)
        K = self.get_k_prime(K, env)
        return np.array(K)

    def get_k_prime(self, K, env):
        # Convert K to ndarray
        K = K.A

        # The internal states of gym envs are different from the internal states of theory, need to reorder the gains
        if env == 'swingup':
            # K := [theta, thetadot]
            # 'Pendulum-v0' gym env states = [cos(theta), sin(theta), thetadot]
            K_prime = [[0, K[0][0], K[0][1]]]

        elif env == 'cartpole':
            # K := [x, x_dot, theta, theta_dot]
            # Cartpole env states = [x, x_dot, np.cos(theta), np.sin(theta), theta_dot]
            K_prime = [[K[0][0], K[0][1], 0, K[0][2], K[0][3]]]

        elif env == 'mountaincar':
            # K := [position, velocity]
            # Mountain car gym env states = [position, velocity]
            K_prime = K

        else:
            logger.error("--- Error: LQR.get_k_prime() env incorrect! ---")
        return K_prime


def calculate_ratio(x, a, S):
    d = (x - a) @ np.diag(S) @ (x - a).transpose()
    ratio = 1 / np.power(d + 1, 2)
    return ratio