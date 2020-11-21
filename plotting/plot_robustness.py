import os
from os import listdir
from typing import List
import numpy as np

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from dao.trainer import Evaluator
from dao.envloader import TFPy2Gym
from plotting.plotter import RobustnessPlotter

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_robustness(envs_names: List[str], envs: List[TFPyEnvironment], policies: List[str],
                   noises: List[float], param_args: List[List[str]]) -> dict:
    all_rewards = {}  # a dict of dict containing awards for all envs

    # Append trajectories by env and policy
    for i, env_name in enumerate(envs_names):
        env_rewards = {}  # a dict containing rewards for the current env
        env_dir = os.path.join('controllers', env_name)

        for policy in policies:
            policy_rewards = None

            foldernames = [f for f in listdir(env_dir) if f.startswith(policy)]
            for model_folder in foldernames:
                print(f'{env_name} {policy} testing {model_folder}')

                # load a controller into memory
                model_path = os.path.join(env_dir, model_folder)
                myEvaluator = Evaluator(eval_env=envs[i], policy=None, plotter=None, model_path=model_path,
                                        eval_num_episodes=1)
                myEvaluator.load_policy()

                for noise in noises:
                    init_position = np.random.uniform(-np.pi * 1 / 180, np.pi * 1 / 180)
                    TFPy2Gym(envs[i]).mutate_with_noise(noise=noise, arg_names=param_args[i], init_position=init_position)
                    myEvaluator(training_time=int(noise * 100), save_model=False)

                data = myEvaluator.get_awardcurve()
                # organise awards for the current policy
                if policy_rewards is None:
                    policy_rewards = data  # policy_rewards is an AwardCurve object for the current policy
                else:
                    policy_rewards.append(data)  # policy_rewards is an AwardCurve object for the current policy

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
    policies = ['ddpg_baseline', 'ddpg_hybrid', 'pilco_baseline', 'pilco_hybrid', 'linear']
    param_args1 = [['m'], ['masscart', 'masspole'], ['masscart']]
    param_args2 = [['g'], ['gravity'], ['gravity']]
    noises = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    all_r1 = get_robustness(envs_names, envs, policies, noises, param_args1)
    all_r2 = get_robustness(envs_names, envs, policies, noises, param_args2)
    all_rewards = [all_r1, all_r2]
    myRobustnessPlotter = RobustnessPlotter(envs_names)
    myRobustnessPlotter(all_rewards)
