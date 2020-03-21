import gpflow
import numpy as np
import control

from pilco.controllers import LinearController


class LinearControllerIPTest(LinearController):
    def __init__(self, A, B, max_action=None, trainable=False):
        super(LinearControllerIPTest, self).__init__(1, 1, max_action=max_action, trainable=trainable)
        self.A = A
        self.B = B
        # control_dim = 1
        # state_dim = len(self.A)

        assert len(self.A) == 2 or len(self.A) == 4, "state_dim wrong?"
        if len(self.A) == 2:
            self.Q = np.array([[1, 0],
                               [0, 0]])
        elif len(self.A) == 4:
            self.Q = np.array([[1, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]])

        self.R = 1

        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        print(type(self.K))

        self.W = gpflow.Param(np.array(self.K), trainable=trainable)
        print(self.W)