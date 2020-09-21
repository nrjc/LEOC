from __future__ import absolute_import, division, print_function

import abc
import numpy as np
import gym
from typing import List

import tensorflow as tf
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, suite_gym, wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class myPendulum(py_environment.PyEnvironment):
    def __init__(self, initialize_top=False):
        self.env = suite_gym.load('Pendulum-v0')
        self._action_spec = self.env._action_spec
        self._observation_spec = self.env._observation_spec
        self._episode_ended = False
        self.up = initialize_top
        self.env.gym.reset = self.modified_reset

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def modified_reset(self):
        if self.up:
            self.env.gym.state = [np.pi / 180, 0.0]
        else:
            self.env.gym.state = [np.pi, 0.0]
        self.env.gym.last_u = None
        self._episode_ended = False
        return self.env.gym._get_obs()

    def _reset(self):
        if self.up:
            self.env.gym.state = [np.pi / 180, 0.0]
        else:
            self.env.gym.state = [np.pi, 0.0]
        self.env.gym.last_u = None
        self._episode_ended = False
        return ts.restart(np.array(self.env.gym.state))

    def _step(self, action):
        return self.env._step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.gym.__dict__[k] = self.env.gym.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of pendulum
        # l := length of pendulum from end to centre
        # b := coefficient of friction of pendulum
        b = 0
        g = self.env.gym.g
        m = self.env.gym.m
        l = self.env.gym.l / 2
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

class myMountainCar(py_environment.PyEnvironment):
    def __init__(self, initialize_top=False):
        self.env = suite_gym.load('Mountaincar-v7')
        self._action_spec = self.env._action_spec
        self._observation_spec = self.env._observation_spec
        self._episode_ended = False
        self.up = initialize_top
        self.env.gym.reset = self.modified_reset

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def modified_reset(self):
        if self.up:
            self.env.state = [np.pi / 180, 0.0]
        else:
            self.env.state = [-np.pi, 0.0]
        self.env.gym.steps_beyond_done = None
        self.env.gym.last_u = None
        self._episode_ended = False
        return self.env.gym._get_obs()

    def _reset(self):
        if self.up:
            self.env.state = [np.pi / 180, 0.0]
        else:
            self.env.state = [-np.pi, 0.0]
        self.env.gym.steps_beyond_done = None
        self.env.gym.last_u = None
        self._episode_ended = False
        return ts.restart(np.array(self.env.gym.state))

    def _step(self, action):
        return self.env._step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.gym.__dict__[k] = self.env.gym.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of car
        # b := coefficient of friction of mountain. 'MountainCarEnv' env is frictionless
        b = 0
        g = self.env.gym.gravity
        m = self.env.gym.masscart

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [g, -b / m]])

        B = np.array([[0],
                      [-1 / m]])

        C = np.array([[1, 0]])

        Q = np.diag([2.0, 0.3])

        return A, B, C, Q


class myCartpole(py_environment.PyEnvironment):
    def __init__(self, initialize_top=False):
        self.env = suite_gym.load('Cartpole-v7')
        self._action_spec = self.env._action_spec
        self._observation_spec = self.env._observation_spec
        self._episode_ended = False
        self.up = initialize_top
        self.env.gym.reset = self.modified_reset

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def modified_reset(self):
        if self.up:
            self.env.state = [0, 0, np.pi / 180, 0]
        else:
            self.env.state = [0.0, 0.0, np.pi, 0.0]
        self.env.gym.steps_beyond_done = None
        self.env.gym.last_u = None
        self._episode_ended = False
        return self.env.gym._get_obs()

    def _reset(self):
        if self.up:
            self.env.state = [0, 0, np.pi / 180, 0]
        else:
            self.env.state = [0.0, 0.0, np.pi, 0.0]
        self.env.gym.steps_beyond_done = None
        self.env.gym.last_u = None
        self._episode_ended = False
        return ts.restart(np.array(self.env.gym.state))

    def _step(self, action):
        return self.env._step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.env.gym.__dict__[k] = self.env.gym.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # M := mass of cart
        # m := mass of pole
        # l := length of pole from end to centre
        # b := coefficient of friction of cart. 'CartPoleEnv' env is frictionless
        b = 0
        M = self.env.gym.masscart
        m = self.env.gym.masspole
        l = self.env.gym.length
        g = self.env.gym.gravity
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
