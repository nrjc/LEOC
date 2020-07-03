import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf

from utils import policy, rollout, Normalised_Env
from examples.envs.mountain_car_env import Continuous_MountainCarEnv

np.random.seed(0)

class myMountainCar():
    def __init__(self):
        self.env = Continuous_MountainCarEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, up=False):
        self.env.reset()
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def control(self):
        A, B, C = None, None, None
        return A, B, C


if __name__ == '__main__':
    # Define params
    test_linear_control = True
    SUBS = 5
    T = 25

    env = myMountainCar()

    if test_linear_control:
        states = env.reset()
        for i in range(100):
            env.render()
            action = [1.0]
            states, _, _, _ = env.step(action)
            print(f'Step: {i}, action: {action}')

    env.close()
