from typing import List
from gym.envs.classic_control import PendulumEnv
from gym import spaces
import numpy as np

float_type = np.float64


class SwingUpEnv(PendulumEnv):
    def __init__(self, init_position=None, name=None):
        super().__init__()
        self.init_position = init_position
        self.target = np.array([1.0, 0.0, 0.0])
        self.tau = self.dt
        self.name = name

        high = np.array([1., 1., self.max_speed], dtype=float_type)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=float_type
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=float_type
        )

    def reset(self):
        if self.init_position is not None:
            self.state = np.array([-np.pi / 180 * self.init_position, 0.0])
        else:
            self.state = np.array([np.pi, 0.0])
            high = np.array([np.pi / 180 * 10, 0.1])
            noise = self.np_random.uniform(low=-high, high=high)
            self.state = self.state + noise
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=float_type)

    def mutate_with_noise(self, noise, arg_names=None, init_position=0.0):
        self.__init__(init_position=init_position)
        if arg_names is None:
            arg_names = ['g']
        for k in arg_names:
            self.__dict__[k] = self.__dict__[k] * (1 + noise)

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
