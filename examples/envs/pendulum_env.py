from typing import List
from gym.envs.classic_control import PendulumEnv
import numpy as np


class SwingUpEnv(PendulumEnv):
    def __init__(self, initialize_top=False, name=None):
        super().__init__()
        self.up = initialize_top
        self.name = name

    def reset(self):
        if self.up:
            self.state = [np.pi / 180, 0.0]
        else:
            self.state = [np.pi, 0.0]
            high = np.array([np.pi/180 * 90, 0.5])
            noise = self.np_random.uniform(low=-high, high=high)
            self.state = self.state + noise
        self.last_u = None
        return self._get_obs()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.__dict__[k] = self.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of pendulum
        # l := length of pendulum from end to centre
        # b := coefficient of friction of pendulum
        b = 0
        g = self.g
        m = self.m
        l = self.l / 2
        I = 1 / 3 * m * (l ** 2)
        p = m * (l ** 2) + I

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [m * l * g / p, -b / p]])

        B = np.array([[0],
                      [-1 / p]])

        C = np.array([[1, 0]])

        Q = np.diag([2.0, 2.0])

        return A, B, C, Q
