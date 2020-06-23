import numpy as np
import gym

from pilco.controller_utils import LQR
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController
from pilco.plotting_utils import plot_single_rollout_cycle
from pilco.rewards import ExponentialReward
from examples.envs.cartpole_env import CartPoleEnv
import tensorflow as tf
from utils import rollout, policy
from matplotlib import pyplot as plt
from gpflow import set_trainable

np.random.seed(0)


class myPendulum():
    def __init__(self):
        self.env = CartPoleEnv() # self.env = gym.make('CartPole-v1').env
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

    def control(self):
        # reference http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        # M := mass of cart
        # m := mass of pole
        # l := length of pole from end to centre
        # b := coefficient of friction of cart, this is 0 in 'InvertedPendulum-v1' env
        b = 0
        M = self.env.masscart
        m = self.env.masspole
        l = self.env.length
        g = self.env.gravity
        I = 1 / 12 * m * (l ** 2)
        p = I * (M + m) + M * m * (l ** 2)

        # using x to approximate sin(x) and 1-x to approximate cos(x)
        A = np.array([[0,                           1,                              0, 0],
                      [0, -(I + m * (l ** 2)) * b / p,  ((m ** 2) * g * (l ** 2)) / p, 0],
                      [0,                           0,                              0, 1],
                      [0,            -(m * l * b) / p,        m * g * l * (M + m) / p, 0]])

        B = np.array([[0],
                      [(I + m * l ** 2) / p],
                      [0],
                      [m * l / p]])

        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        return A, B, C


# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

if __name__ == '__main__':
    SUBS = 3
    bf = 60
    maxiter = 50
    max_action = 2.0
    target = np.array([1.0, 0.0, 0.0])
    weights = np.diag([2.0, 2.0, 0.3])
    m_init = np.reshape([-1.0, 0, 0.0], (1, 3))
    S_init = np.diag([0.01, 0.05, 0.01])
    T = 40
    T_sim = T
    J = 5
    N = 8
    restarts = 2

    # env = myPendulum()
    # A, B, C = env.control()
    # W_matrix = LQR().get_W_matrix(A, B, C, env='cartpole')
    #
    # # Set up objects and variables
    # state_dim = env.observation_space.shape[0]
    # control_dim = 1
    # # In 'CartPole-v1', control actions are discrete
    # controls_discrete = {'left': 0, 'right': 1}
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim, W=-W_matrix, max_action=1.0)
    # # controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
    # #                                 controller_location=target, W=-W_matrix, max_action=2.0)
    # R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    #
    # # Initial random rollouts to generate a dataset
    # X, Y, _, _ = rollout(env=env, pilco=None, timesteps=T, random=True, SUBS=SUBS, render=True, verbose=False)
    # for i in range(1, J):
    #     X_, Y_, _, _ = rollout(env=env, pilco=None, timesteps=T, random=True, SUBS=SUBS, render=True, verbose=False)
    #     X = np.vstack((X, X_))
    #     Y = np.vstack((Y, Y_))
    # pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)
    # np.random.seed(0)

    # state_dim = Y.shape[1]
    # control_dim = X.shape[1] - state_dim
    # controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # # controller = LinearController(state_dim=state_dim, control_dim=control_dim)
    #
    # pilco = PILCO(X, Y, controller=controller, horizon=40)
    # # Example of user provided reward function, setting a custom target state
    # # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)
    #
    # # Example of fixing a parameter, optional, for a linear controller only
    # # pilco.controller.b = np.array([[0.0]])
    # # pilco.controller.b.trainable = False
    #
    # for rollouts in range(3):
    #     pilco.optimize_models()
    #     pilco.optimize_policy()
    #     import pdb
    #
    #     pdb.set_trace()
    #     X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=100)
    #     print("No of ops:", len(tf.get_default_graph().get_operations()))
    #     # Update dataset
    #     X = np.vstack((X, X_new))
    #     Y = np.vstack((Y, Y_new))
    #     pilco.mgpr.set_XY(X, Y)

    env = CartPoleEnv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()