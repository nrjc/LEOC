import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import CombinedController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging

np.random.seed(0)
# TODO: Try with Combined Controller
from utils import rollout, policy


class myPendulum():
    def __init__(self):
        self.env = gym.make('InvertedPendulum-v2').env
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

    def ControllerInverted(self):
        # reference http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        # M = mass of cart
        # m = mass of pendulum
        # l = length of pendulum
        # b = coefficient of friction of cart
        b = 0
        M = self.env.masscart
        m = self.env.masspole
        l = self.env.length * 2
        g = self.env.gravity
        I = 1 / 12 * m * l ** 2
        r = l / 2
        p = I * (M + m) + M * m * r ** 2

        A = np.array([[0, 1, 0, 0],
                      [0, -(I + m * r ** 2) * b / p, (m ** 2 * g * r ** 2) / p, 0],
                      [0, 0, 0, 1],
                      [0, -(m * r * b) / p, m * g * r * (M + m) / p, 0]])

        B = np.array([[0],
                      [(I + m * r ** 2) / p],
                      [0],
                      [m * r / p]])

        return A, B

    def render(self):
        self.env.render()


with tf.Session(graph=tf.Graph()) as sess:
    env = myPendulum()
    # Initial random rollouts to generate a dataset
    X, Y = rollout(env=env, pilco=None, random=True, timesteps=40)
    for i in range(1, 3):
        X_, Y_ = rollout(env=env, pilco=None, random=True, timesteps=40)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    pilco = PILCO(X, Y, controller=controller, horizon=40)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    # Example of fixing a parameter, optional, for a linear controller only
    # pilco.controller.b = np.array([[0.0]])
    # pilco.controller.b.trainable = False

    for rollouts in range(3):
        pilco.optimize_models()
        pilco.optimize_policy()
        import pdb;

        pdb.set_trace()
        X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=100)
        print("No of ops:", len(tf.get_default_graph().get_operations()))
        # Update dataset
        X = np.vstack((X, X_new));
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)
