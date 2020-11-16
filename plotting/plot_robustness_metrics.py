import os
from os import listdir
from typing import List
import numpy as np

import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from dao.trainer import Evaluator
from dao.envloader import TFPy2Gym
from plotting.plotter import RobustnessPlotter, AwardCurve

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_metrics(envs_names: List[str], envs: List[TFPyEnvironment], policies: List[str], noises: List[float]) -> dict:
    all_rewards = {}  # a dict of dict containing awards for all envs

    # Append trajectories by env and policy
    for i, env_name in enumerate(envs_names):
        env_rewards = {}  # a dict containing rewards for the current env
        env_dir = os.path.join('controllers', env_name)

        for policy in policies:
            policy_rewards = None
            model_path = os.path.join(env_dir, policy)
            myEvaluator = Evaluator(eval_env=envs[i], policy=None, plotter=None, model_path=model_path, eval_num_episodes=1)
            myEvaluator.load_policy()

            for noise in noises:
                myEvaluator.eval_results = []

                for _ in range(1):
                    # noise injection get data points for each (env, policy, noise) setting
                    init_position = np.random.uniform(-np.pi * 10/180, np.pi * 10/180)
                    TFPy2Gym(envs[i]).mutate_with_noise(noise=noise, init_position=init_position)
                    myEvaluator(training_timesteps=0, save_model=False)

                x = np.array([noise])
                y = np.array(myEvaluator.eval_results).reshape((1, 1))
                data = np.vstack((x, y))
                data = AwardCurve(data)

                # organise awards for the current policy
                if policy_rewards is None:
                    policy_rewards = data  # policy_rewards is an AwardCurve object for the current policy
                else:
                    policy_rewards.h_append(data)  # policy_rewards is an AwardCurve object for the current policy

            env_rewards[policy] = policy_rewards
        all_rewards[env_name] = env_rewards
    return all_rewards


if __name__ == "__main__":
    pendulum_py_env = suite_gym.load('Pendulum-v8')
    pendulum_env = tf_py_environment.TFPyEnvironment(pendulum_py_env)
    cartpole_py_env = suite_gym.load('Cartpole-v8')
    cartpole_env = tf_py_environment.TFPyEnvironment(cartpole_py_env)
    mountaincar_py_env = suite_gym.load('Mountaincar-v8')
    mountaincar_env = tf_py_environment.TFPyEnvironment(mountaincar_py_env)

    envs = [pendulum_env, cartpole_env, mountaincar_env]
    envs_names = ['Pendulum', 'Cartpole', 'Mountaincar']
    policies = ['ddpg_baseline0', 'pilco_baseline0', 'linear']
    noises = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    all_rewards = get_metrics(envs_names, envs, policies, noises)
    myMetricsPlotter = RobustnessPlotter(envs_names)
    myMetricsPlotter(all_rewards)
