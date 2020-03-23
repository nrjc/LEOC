import gpflow
import numpy as np
import control

from pilco.controllers import LinearController


class LinearControllerTest(LinearController):
    def __init__(self, A, B, C, max_action=None, trainable=False):
        super(LinearControllerTest, self).__init__(1, 1, max_action=max_action, trainable=trainable)
        self.A = A
        self.B = B
        self.C = C
        # control_dim = 1
        # state_dim = len(self.A)

        assert len(self.A) == 2 or len(self.A) == 4, "state_dim wrong?"
        self.Q = self.C.T * self.C
        self.R = 1

        K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        self.K = self.get_k_prime(K)
        print(self.K)

        self.W = gpflow.Param(np.array(self.K), trainable=trainable)
        print(self.W)

    def get_k_prime(self, K, swing_up=1):
        K = K.A
        if swing_up:
            k_prime = [[0, K[0][0], K[0][1]]]
        return k_prime