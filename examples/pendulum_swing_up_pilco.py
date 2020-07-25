import logging

logging.basicConfig(level=logging.INFO)
import numpy as np
import gym
import random
from pilco.controller_utils import LQR
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController
from pilco.plotting_utils import plot_single_rollout_cycle
from pilco.rewards import ExponentialReward, L2HarmonicPenalization, CombinedRewards
import tensorflow as tf
from utils import rollout, policy
from matplotlib import pyplot as plt
from gpflow import set_trainable

np.random.seed(0)


# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

class myPendulum():
    def __init__(self, initialize_top=False):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.up = initialize_top

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01 * high)  # only difference
        if not self.up:
            self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

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

        return A, B, C


if __name__ == '__main__':
    # Define params
    test_linear_control = False
    SUBS = 3
    bf = 30
    maxiter = 50
    max_action = 1.0
    target = np.array([1.0, 0.0, 0.0])
    weights = np.diag([2.0, 2.0, 0.3])
    m_init = np.reshape([-1.0, 0, 0.0], (1, 3))
    S_init = np.diag([0.01, 0.05, 0.01])
    T = 40
    J = 4
    N = 8
    restarts = 2

    # Set up objects and variables
    env = myPendulum(False)
    A, B, C = env.control()
    W_matrix = LQR().get_W_matrix(A, B, C, env='swing up')

    state_dim = 3
    control_dim = 1
    controller_linear = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix, max_action=max_action)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    # controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
    #                                 controller_location=target, W=W_matrix, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    if not test_linear_control:
        # Initial random rollouts to generate a dataset
        X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=False, verbose=False)
        for i in range(1, J):
            X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=False, verbose=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

        # for numerical stability, we can set the likelihood variance parameters of the GP models
        for model in pilco.mgpr.models:
            model.likelihood.variance.assign(0.001)
            set_trainable(model.likelihood.variance, False)

        r_new = np.zeros((T, 1))
        all_rewards = []

        for rollouts in range(N):
            print("**** ITERATION no", rollouts, " ****")
            policy_restarts = 1 if rollouts > 3 else 2
            pilco.optimize_models(maxiter=maxiter, restarts=restarts)
            pilco.optimize_policy(maxiter=maxiter, restarts=policy_restarts)
            s_val = pilco.get_controller()
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T, verbose=False, SUBS=SUBS)

            # Since we had decide on the various parameters of the reward function
            # we might want to verify that it behaves as expected by inspection
            for i in range(len(X_new)):
                r_new[:, 0] = R.compute_reward(X_new[i, None, :-1], 0.001 * np.eye(state_dim))[0]
            total_r = sum(r_new)
            _, _, r, intermediary_dict = pilco.predict_and_obtain_intermediates(X_new[0, None, :-1], 0.001 * S_init, T)
            print("Total ", total_r, " Predicted: ", r)

            # Plotting internal states of pilco variables
            intermediate_mean, intermediate_var, intermediate_reward = zip(*intermediary_dict)
            intermediate_var = [x.diagonal() for x in intermediate_var]
            intermediate_mean = [x[0] for x in intermediate_mean]
            # get reward of the last time step
            rollout_reward = intermediate_reward[T - 1][0]
            rollout_reward = np.array(rollout_reward)
            all_rewards.append(rollout_reward[0])
            plot_single_rollout_cycle(intermediate_mean, intermediate_var, [X_new], None, all_rewards, state_dim,
                                      control_dim, T, rollouts, env='swing up')

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
        plt.show()

    else:
        env.up = True
        states = env.reset()
        for i in range(100):
            env.render()
            action = controller_linear.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                               tf.zeros([state_dim, state_dim], dtype=tf.dtypes.float64),
                                               squash=True)[0]
            action = action[0, :].numpy()
            states, _, _, _ = env.step(action)
            print(f'Step: {i}, action: {action}')

    env.close()
