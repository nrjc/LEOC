import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
from pilco.envs_utils import myCartpole
from pilco.models import PILCO
from pilco.controllers import LinearController, CombinedController
from controller_utils import LQR, calculate_ratio
from pilco.plotting_utils import plot_single_rollout_cycle
from pilco.rewards import ExponentialReward
from pilco.utils import rollout, save_gpflow_obj_to_path
import os
np.random.seed(0)


if __name__ == '__main__':
    # Define params
    test_linear_control = False
    SUBS = 3
    bf = 60
    maxiter = 50
    max_action = 2.5
    T = 40
    J = 5
    N = 12
    restarts = 2
    model_save_dir = './'

    # States := [x, x_dot, cos(theta), sin(theta), theta_dot]
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    weights = np.diag([2.0, 0.3, 2.0, 2.0, 0.3])
    m_init = np.reshape([0.0, 0.0, -1.0, 0.0, 0.0], (1, 5))
    S_init = np.diag([0.01, 0.01, 0.01, 0.05, 0.01])

    # Set up objects and variables
    env = myCartpole(False)
    A, B, C, Q = env.control()
    W_matrix = LQR().get_W_matrix(A, B, Q, env='cartpole')

    state_dim = 5 # state_dim = env.observation_space.shape[0]
    control_dim = 1

    controller_linear = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix, max_action=max_action)
    # controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
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

        reward_new = np.zeros((T, 1))
        all_rewards = []
        all_S = np.empty((0, state_dim), float)

        for rollouts in range(N):
            print("**** ITERATION no", rollouts, " ****")
            policy_restarts = 1 if rollouts > 3 else 2
            pilco.optimize_models(maxiter=maxiter, restarts=restarts)
            pilco.optimize_policy(maxiter=maxiter, restarts=policy_restarts)
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T, verbose=False, SUBS=SUBS)

            # Since we had decide on the various parameters of the reward function
            # we might want to verify that it behaves as expected by inspection
            for i in range(len(X_new)):
                reward_new[:, 0] = R.compute_reward(X_new[i, None, :-1], 0.001 * np.eye(state_dim))[0]
            total_reward = sum(reward_new)
            _, _, reward, intermediary_dict = pilco.predict_and_obtain_intermediates(X_new[0, None, :-1], 0.001 * S_init, T)
            print("Total ", total_reward, " Predicted: ", reward)

            # Plotting internal states of pilco variables
            intermediate_mean, intermediate_var, intermediate_reward = zip(*intermediary_dict)
            intermediate_var = [x.diagonal() for x in intermediate_var]
            intermediate_mean = [x[0] for x in intermediate_mean]
            # Get reward of the last time step
            rollout_reward = intermediate_reward[T - 1][0]
            rollout_reward = np.array(rollout_reward)
            all_rewards.append(rollout_reward[0])
            # Get S of the rollout
            rollout_S = pilco.get_controller().S.read_value().numpy()
            rollout_S_inverse = np.array([[1 / lam for lam in rollout_S]])
            all_S = np.append(all_S, rollout_S_inverse, axis=0)
            # Get linear controller ratio for each timestep of the rollout
            realised_states = [x[:state_dim] for x in X_new]
            rollout_ratio = [calculate_ratio(x, target, rollout_S) for x in realised_states]

            # write_to_csv = True if rollouts >= N - 3 else False
            write_to_csv = False
            plot_single_rollout_cycle(intermediate_mean, intermediate_var, [X_new], None, all_rewards, all_S,
                                      rollout_ratio, state_dim, control_dim, T, rollouts, env='cartpole',
                                      write_to_csv=write_to_csv)

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
            save_gpflow_obj_to_path(controller, os.path.join(model_save_dir, f'cartpole_controller{rollouts}.pkl'))
        plt.show()

    else:
        env.up = True
        states = env.reset()
        for i in range(500):
            env.render()
            action = controller_linear.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                                      tf.zeros([state_dim, state_dim], dtype=tf.dtypes.float64),
                                                      squash=False)[0]
            action = action[0, :].numpy()
            states, _, _, _ = env.step(action)
            print(f'Step: {i}, action: {action}')

    env.close()