import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController
from pilco.controller_utils import LQR
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
from utils import rollout, policy
import random

np.random.seed(0)


# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # high = np.array([np.pi, 1])
        # self.env.state = np.random.uniform(low=-high, high=high)
        # self.env.state = np.random.uniform(low=0, high=0.01 * high)  # only difference
        # self.env.state[0] += -np.pi
        self.env.state = np.array([random.normalvariate(0, 0.2), 0])
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def control(self):
        # m := mass of pendulum
        # l := length of pendulum
        # b := coefficient of friction of pendulum
        b = 0
        g = self.env.g
        m = self.env.m
        l = self.env.l
        I = 1 / 12 * m * (l ** 2)
        p = 1 / 4 * m * (l ** 2) + I

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [1 / 2 * m * l * g / p, -b / p]])

        B = np.array([[0],
                      [1 / p]])

        C = np.array([[1, 0]])

        return A, B, C


SUBS = 3
bf = 30
maxiter = 50
max_action = 2.0
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])
m_init = np.reshape([-1.0, 0, 0.0], (1, 3))
S_init = np.diag([0.01, 0.05, 0.01])
T = 40
T_sim = T
J = 100
N = 1
restarts = 2

with tf.Session() as sess:
    env = myPendulum()

    A, B, C = env.control()
    W_matrix = LQR().get_W_matrix(A, B, C)
    state_dim = env.observation_space.shape[0]
    control_dim = env.action_space.shape[0]
    # controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, W=W_matrix, max_action=max_action)
    controller = LinearController(state_dim=state_dim, control_dim=control_dim, W=-W_matrix, max_action=max_action,
                                  trainable=False)
    controller.b = np.zeros([1, control_dim])

    states = env.reset()
    for i in range(J):
        env.render()
        action = controller.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                           tf.zeros([state_dim, state_dim], dtype=tf.dtypes.float64),
                                           squash=False)[0]
        action = action.eval()[0, :] + random.normalvariate(0, 0.1)
        states, _, _, _ = env.step(action)
        print(f'Step: {i}')
