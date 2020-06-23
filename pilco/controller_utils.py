import gpflow
import numpy as np
import control

from pilco.controllers import LinearController


class LQR:
    def get_W_matrix(self, A, B, C, env='swing up'):
        Q = np.dot(C.T, C)
        R = 1
        K, _, _ = control.lqr(A, B, Q, R)
        K = self.get_k_prime(K, env)
        return np.array(K)

    def get_k_prime(self, K, env='swing up'):
        # reference http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace#6
        assert (env == 'swing up' or env == 'cartpole'), "--- LQR.get_k_prime() error! ---"

        # Convert K to ndarray
        K = K.A

        # The internal states of gym envs are different from the internal states of theory, need to reorder the gains
        if env == 'swing up':
            # K := [theta, thetadot]
            # 'Pendulum-v0' gym env states = [cos(theta), sin(theta), thetadot]
            K_prime = [[0, K[0][0], K[0][1]]]

        elif env == 'cartpole':
            # K := [x, xdot/vel, theta/pos, thetadot]
            # 'InvertedPendulum-v1' gym env states = [pos, vel]
            K_prime = [[K[0][2], K[0][1]]]
        return K_prime