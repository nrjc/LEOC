import gpflow
import numpy as np
import control

from pilco.controllers import LinearController


class LQR:
    def get_W_matrix(self, A, B, C):
        Q = C.T * C
        R = 1
        K, _, _ = control.lqr(A, B, Q, R)
        K = self.get_k_prime(K)
        return np.array(K)

    def get_k_prime(self, K, swing_up=1):
        K = K.A
        if swing_up:
            k_prime = [[0, K[0][0], K[0][1]]]
        return k_prime