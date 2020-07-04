import numpy as np
import gym

from pilco.controller_utils import LQR
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController
from pilco.plotting_utils import plot_single_rollout_cycle
from pilco.rewards import ExponentialReward
from examples.envs.mountain_car_env import Continuous_MountainCarEnv
import tensorflow as tf
from utils import rollout, policy
from matplotlib import pyplot as plt
from gpflow import set_trainable

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
        high = np.array([1, 0])
        self.env.state += np.random.uniform(low=-0.5 * high, high=0.5 * high)
        if up:
            self.env.state[0] += np.pi
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def control(self):
        # m := mass of cart
        # b := coefficient of friction of pendulum
        b = 0 # assume frictionless system
        g = self.env.gravity
        m = self.env.masscart

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [g, -b / m]])

        B = np.array([[0],
                      [-1 / m]])

        C = np.array([[1, 0]])

        return A, B, C


if __name__ == '__main__':
    # Define params
    test_linear_control = True
    SUBS = 5
    T = 25
    max_action = 50.0

    env = myMountainCar()
    A, B, C = env.control()
    W_matrix = LQR().get_W_matrix(A, B, C, env='mountain car')

    # Set up objects and variables
    state_dim = 2
    control_dim = 1
    controller = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix, max_action=max_action)

    if test_linear_control:
        states = env.reset(up=test_linear_control)
        for i in range(200):
            env.render()
            action = controller.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                               tf.zeros([state_dim, state_dim], dtype=tf.dtypes.float64),
                                               squash=False)[0]
            action = action[0, :].numpy()
            states, _, _, _ = env.step(action)
            print(f'Step {i}: action={action}; x={states[0]:.2f}; x_dot={states[1]:.3f}')

    env.close()
