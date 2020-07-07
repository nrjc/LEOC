import numpy as np
import gym

from pilco.controller_utils import LQR
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, CombinedController
from pilco.plotting_utils import plot_single_rollout_cycle
from pilco.rewards import ExponentialReward
from examples.envs.mountain_car_env import Continuous_MountainCarEnv as MountainCarEnv
import tensorflow as tf
from utils import rollout, policy
from matplotlib import pyplot as plt
from gpflow import set_trainable

np.random.seed(0)

class myMountainCar():
    def __init__(self):
        self.env = MountainCarEnv()
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

        return A, B, C


if __name__ == '__main__':
    # Define params
    test_linear_control = False
    SUBS = 3
    bf = 60
    maxiter = 50
    max_action = 50.0
    T = 40
    J = 5
    N = 8
    restarts = 2

    # Need to double check init values
    # States := [x, x_dot]
    target = np.array([0.0, 0.0])
    weights = np.diag([2.0, 0.3])
    m_init = np.reshape([-np.pi, 0.0], (1, 2))
    S_init = np.diag([0.01, 0.01])

    env = myMountainCar()
    A, B, C = env.control()
    W_matrix = LQR().get_W_matrix(A, B, C, env='mountain car')

    # Set up objects and variables
    state_dim = 2
    control_dim = 1
    controller_linear = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix,
                                         max_action=max_action)
    controller = CombinedController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                                    controller_location=target, W=W_matrix, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    if not test_linear_control:
        # Initial random rollouts to generate a dataset
        X, Y, _, _ = rollout(env=env, pilco=None, timesteps=T, random=True, SUBS=SUBS, render=True, verbose=False)
        for i in range(1, J):
            X_, Y_, _, _ = rollout(env=env, pilco=None, timesteps=T, random=True, SUBS=SUBS, render=True, verbose=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
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
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T, verbose=False, SUBS=SUBS, render=True)

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
            intermediate_mean, intermediate_var, intermediate_reward = zip(*intermediary_dict)
            intermediate_var = [x.diagonal() for x in intermediate_var]
            intermediate_mean = [x[0] for x in intermediate_mean]
            plot_single_rollout_cycle(intermediate_mean, intermediate_var, [X_new], None, None, state_dim,
                                      control_dim, T, rollouts + 1, env='mountain car')

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
        plt.show()

    else:
        states = env.reset(up=test_linear_control)
        for i in range(200):
            env.render()
            action = controller_linear.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                               tf.zeros([state_dim, state_dim], dtype=tf.dtypes.float64),
                                               squash=False)[0]
            action = action[0, :].numpy()
            states, _, _, _ = env.step(action)
            print(f'Step {i}: action={action}; x={states[0]:.2f}; x_dot={states[1]:.3f}')

    env.close()
