from typing import List

import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from dao.trainer import Evaluator
from plotting.plotter import ControlMetricsPlotter

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_metrics(envs_names: List[str], envs: List[TFPyEnvironment], policies: List[List[str]]) -> dict:
    all_trajectories = {}  # a dict of dict containing trajectories for all envs

    # Append trajectories by env and policy
    for i in range(len(envs_names)):
        env_trajectories = {}  # a dict containing trajectories for the current env

        for policy in policies:
            path = '../controllers/' + policy[i]
            myEvaluator = Evaluator(eval_env=envs[i], policy=None, plotter=None, model_path=path, eval_num_episodes=1)
            myEvaluator.load_policy()
            trajectory = myEvaluator(training_timesteps=0, save_model=False,
                                     impulse_input=-1.0)  # impulse_input: float = 0.0,
            env_trajectories[policy[i]] = trajectory

        all_trajectories[envs_names[i]] = env_trajectories

    return all_trajectories


if __name__ == "__main__":
    pendulum_py_env = suite_gym.load('Pendulum-v8')
    pendulum_env = tf_py_environment.TFPyEnvironment(pendulum_py_env)
    cartpole_py_env = suite_gym.load('Cartpole-v8')
    cartpole_env = tf_py_environment.TFPyEnvironment(cartpole_py_env)
    mountaincar_py_env = suite_gym.load('Mountaincar-v8')
    mountaincar_env = tf_py_environment.TFPyEnvironment(mountaincar_py_env)
    ddpg_baseline = ['Pendulum/ddpg_baseline1', 'Cartpole/ddpg_baseline2_-57', 'Mountaincar/ddpg_baseline3_-124']
    ddpg_hybrid = ['Pendulum/ddpg_hybrid1', 'Cartpole/ddpg_hybrid1_-60', 'Mountaincar/ddpg_hybrid2']
    pilco_baseline = ['Pendulum/pilco_baseline3', 'Cartpole/pilco_baseline1', 'Mountaincar/pilco_baseline3']
    pilco_hybrid = ['Pendulum/pilco_hybrid3', 'Cartpole/pilco_hybrid3', 'Mountaincar/pilco_hybrid2']
    linear_ctrl = ['Pendulum/linear', 'Cartpole/linear', 'Mountaincar/linear']

    py_envs = [pendulum_py_env, cartpole_py_env, mountaincar_py_env]
    envs_names = [py_env.unwrapped.spec.id[:-3] for py_env in py_envs]
    envs = [pendulum_env, cartpole_env, mountaincar_env]
    policies = [ddpg_baseline, ddpg_hybrid, pilco_baseline, pilco_hybrid]

    trajectories = get_metrics(envs_names, envs, policies)
    myMetricsPlotter = ControlMetricsPlotter(py_envs)
    myMetricsPlotter(trajectories)
