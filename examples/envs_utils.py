from typing import List

import gym
import numpy as np

from examples.envs.cartpole_env import CartPoleEnv
from examples.envs.mountain_car_env import Continuous_MountainCarEnv as MountainCarEnv


class myPendulum():
    def __init__(self, initialize_top=False):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space_dim = self.observation_space.shape[0]
        self.up = initialize_top

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.up:
            self.env.state = [np.pi / 180, 0]
        else:
            self.env.state = [np.pi, 0]
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.__dict__[k] = self.env.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of pendulum
        # l := length of pendulum from end to centre
        # b := coefficient of friction of pendulum
        b = 0
        g = self.env.g
        m = self.env.m
        l = self.env.l / 2
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


class myMountainCar():
    def __init__(self, initialize_top=False):
        self.env = MountainCarEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space_dim = self.observation_space.shape[0]
        self.up = initialize_top

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.up:
            self.env.state = [np.pi / 180, 0.0]
        else:
            self.env.state = [-np.pi, 0.0]
        self.env.steps_beyond_done = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.__dict__[k] = self.env.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of car
        # b := coefficient of friction of mountain. 'MountainCarEnv' env is frictionless
        b = 0
        g = self.env.gravity
        m = self.env.masscart

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [g, -b / m]])

        B = np.array([[0],
                      [-1 / m]])

        C = np.array([[1, 0]])

        Q = np.diag([2.0, 0.3])

        return A, B, C, Q


class myCartpole():
    def __init__(self, initialize_top=False):
        self.env = CartPoleEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space_dim = self.observation_space.shape[0] + 1
        self.up = initialize_top

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.up:
            self.env.state = [0, 0, 0, 0]
        else:
            self.env.state = [0.0, 0.0, np.pi, 0.0]
        self.env.steps_beyond_done = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.__dict__[k] = self.env.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # Reference http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        # Reference https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
        # M := mass of cart
        # m := mass of pole
        # l := length of pole from end to centre
        # b := coefficient of friction of cart. 'CartPoleEnv' env is frictionless
        b = 0
        M = self.env.masscart
        m = self.env.masspole
        l = self.env.length
        g = self.env.gravity
        I = 1 / 3 * m * (l ** 2)
        p = I * (M + m) + M * m * (l ** 2)

        # using x to approximate sin(x) and 1 to approximate cos(x)
        A = np.array([[0, 1, 0, 0],
                      [0, -(I + m * (l ** 2)) * b / p, (m ** 2) * g * (l ** 2) / p, 0],
                      [0, 0, 0, 1],
                      [0, (m * l * b) / p, m * g * l * (M + m) / p, 0]])

        B = np.array([[0],
                      [-(I + m * (l ** 2)) / p],
                      [0],
                      [m * l / p]])

        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        Q = np.diag([2.0, .3, 2.0, 0.3])

        return A, B, C, Q