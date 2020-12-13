import os
from os import listdir
from typing import List

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from dao.trainer import Evaluator
from plotting.plotter import TransientResponsePlotter

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_directory(env_dir: str, eval_env: TFPyEnvironment, policy: str, impulse_input=0.0, step_input=0.0):
    policy_trajectories = []
    foldernames = [f for f in listdir(env_dir) if f.startswith(policy)]
    for model_folder in foldernames:
        model_path = os.path.join(env_dir, model_folder)
        myEvaluator = Evaluator(eval_env=eval_env, policy=None, load_model_path=model_path)
        myEvaluator.load_policy()
        trajectory = myEvaluator(training_time=0, save_model=False, impulse_input=impulse_input, step_input=step_input)
        policy_trajectories.append(trajectory)

    return policy_trajectories


def get_response(envs_names: List[str], envs: List[TFPyEnvironment], policies: List[str],
                impulse_input=0.0, step_input=0.0) -> dict:
    all_trajectories = {}  # a dict of dict containing trajectories for all envs

    # Append trajectories by env and policy
    for i, env_name in enumerate(envs_names):
        env_trajectories = {}  # a dict containing trajectories for the current env
        env_dir = os.path.join('controllers', env_name)

        for policy in policies:
            policy_trajectories = load_directory(env_dir, envs[i], policy, impulse_input=impulse_input,
                                                 step_input=step_input)

            env_trajectories[policy] = policy_trajectories
        all_trajectories[env_name] = env_trajectories

    return all_trajectories


if __name__ == "__main__":
    pendulum_py_env = suite_gym.load('Pendulum-v8')
    pendulum_env = tf_py_environment.TFPyEnvironment(pendulum_py_env)
    cartpole_py_env = suite_gym.load('Cartpole-v8')
    cartpole_env = tf_py_environment.TFPyEnvironment(cartpole_py_env)
    mountaincar_py_env = suite_gym.load('Mountaincar-v8')
    mountaincar_env = tf_py_environment.TFPyEnvironment(mountaincar_py_env)

    envs = [pendulum_env, cartpole_env, mountaincar_env]
    envs_names = ['Pendulum', 'Cartpole', 'Mountaincar']
    policies = ['ddpg_baseline0', 'ddpg_hybrid0', 'pilco_baseline0', 'pilco_hybrid0', 'linear']

    impulse_trajectories = get_response(envs_names, envs, policies, impulse_input=-1.0)
    step_trajectories = get_response(envs_names, envs, policies, step_input=-0.2)
    myMetricsPlotter = TransientResponsePlotter(envs_names)
    myMetricsPlotter([impulse_trajectories, step_trajectories])
