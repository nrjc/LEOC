import numpy as np
import gym

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


# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts


if __name__ == '__main__':
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

    env = myPendulum()
    A, B, C = env.control()
    W_matrix = LQR().get_W_matrix(A, B, C)

    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=True)
    for i in range(1, J):
        X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                                    controller_location=np.array([[0, 1, 0]], dtype=np.float64), max_action=max_action,
                                    W=-W_matrix)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    c_param = L2HarmonicPenalization([controller.get_S()], 0.0001)
    R = CombinedRewards(state_dim, [R, c_param])

    pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability, we can set the likelihood variance parameters of the GP models
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)
    axis_values = np.zeros((N, state_dim))
    r_new = np.zeros((T, 1))
    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)
        s_val = pilco.get_controller().get_S()
        axis_values[rollouts, :] = s_val.numpy()
        X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS, render=True)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        for i in range(len(X_new)):
            r_new[:, 0] = R.compute_reward(X_new[i, None, :-1], 0.001 * np.eye(state_dim))[0]
        total_r = sum(r_new)
        _, _, r, intermediary_dict = pilco.predict_and_obtain_intermediates(X_new[0, None, :-1], 0.001 * S_init, T)
        print("Total ", total_r, " Predicted: ", r)
        plt.figure(3)
        for c_dim in range(state_dim):
            plt.plot(axis_values[:rollouts, c_dim])
        plt.pause(0.01)
        # Plotting internal states of pilco variables
        intermediate_mean, intermediate_var = zip(*intermediary_dict)
        intermediate_var = [x.diagonal() for x in intermediate_var]
        intermediate_mean = [x[0] for x in intermediate_mean]
        plot_single_rollout_cycle(intermediate_mean, intermediate_var, [X_new], None, state_dim, control_dim, T, 1)

        # Update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
    plt.show()
