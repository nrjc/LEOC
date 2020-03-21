import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController, ControllerSwingUp
from pilco.controller_utils import LinearControllerIPTest
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
from utils import rollout, policy

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
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01 * high)  # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def ControllerSwingUp(self):
        # m = mass of pendulum
        # l = length of pendulum
        # b = coefficient of friction of pendulum
        b = 0
        g = self.env.g
        m = self.env.m
        l = self.env.l
        I = 1 / 12 * m * l ** 2
        p = 1 / 4 * m * l ** 2 + I

        # using x to approximate sin(x)
        A = np.array([[-b / p, -1 / 2 * m * l * g],
                      [1, 0]])

        B = np.array([[1 / p],
                      [0]])

        return A, B


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
J = 4
N = 8
restarts = 2

with tf.Session() as sess:
    env = myPendulum()

    # Initial random rollouts to generate a dataset
    X, Y = rollout(env, None, timesteps=T, random=True, SUBS=SUBS)
    for i in range(1, J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    # controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    A, B = env.ControllerSwingUp()
    controller = LinearControllerIPTest(A, B)

    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # cur_rew = 0
        # for t in range(0,len(X_new)):
        #     cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        # print('On this episode reward was ', cur_rew)

        # Update dataset
        X = np.vstack((X, X_new));
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)
