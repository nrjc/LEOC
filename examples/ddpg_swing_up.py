import numpy as np
import logging

import tensorflow as tf
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import random_tf_policy

from examples.envs.pendulum_env import SwingUpEnv
from DDPG.ddpg import DDPG, LinearController, ReplayBuffer, train_agent
from controller_utils import LQR

logging.basicConfig(level=logging.INFO)
from utils import load_controller_from_obj
import os

np.random.seed(0)

if __name__ == '__main__':
    # Define params
    test_linear_control = False
    num_iterations = 20000  # @param {type:"integer"}
    initial_collect_steps = 1000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 100000  # @param {type:"integer"}
    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 10  # @param {type:"integer"}
    num_eval_episodes = 5  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    target = np.array([1.0, 0.0, 0.0])

    train_py_env = suite_gym.load('Pendulum-v7')
    eval_py_env = suite_gym.load('Pendulum-v7')
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    A, B, C, Q = train_py_env.control()
    W_matrix = LQR().get_W_matrix(A, B, Q, env='swing up')

    state_dim = 3
    control_dim = 1

    controller_linear = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix)

    if not test_linear_control:
        # print('action_spec:', env._action_spec)
        # print('observation_spec:', env._observation_spec)
        # print('time_step_spec.step_type:', env.time_step_spec().step_type)
        # print('time_step_spec.discount:', env.time_step_spec().discount)
        # print('time_step_spec.reward:', env.time_step_spec().reward)

        myDDPGagent = DDPG(train_env, linear_controller=controller_linear, controller_location=target)
        myReplayBuffer = ReplayBuffer(myDDPGagent, train_env, replay_buffer_capacity, initial_collect_steps,
                                      collect_steps_per_iteration)

        # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
        train_agent(myDDPGagent,
                    myReplayBuffer,
                    eval_env=eval_env,
                    num_iterations=num_iterations,
                    initial_collect_steps=initial_collect_steps,
                    collect_steps_per_iteration=collect_steps_per_iteration,
                    num_eval_episodes=num_eval_episodes,
                    log_interval=log_interval,
                    eval_interval=eval_interval)

    else:
        # controller_path = os.path.join(model_save_dir, 'controllers', 'swingup_rbf_controller4.pkl')
        # controller = load_controller_from_obj(controller_path)

        train_py_env.gym.up = True
        train_py_env.reset()
        state = train_py_env.gym._get_obs()

        for i in range(100):
            train_py_env.render()
            action = controller_linear.compute_action(tf.reshape(tf.convert_to_tensor(state), (1, -1)))[0]
            timestep = train_py_env.step(action)
            state = timestep.observation
            print(f'Step: {i}, state: {state}')

    train_py_env.close()
