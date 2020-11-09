from typing import Dict, List

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories.trajectory import Trajectory

import dao.envs
from dao.trainer import Evaluator
from plotting.plotter import ControlMetricsPlotter
import numpy as np

ddpg_baseline = ['pendulum/ddpg_baseline1', 'cartpole/ddpg_baseline2_-57', 'mountaincar/ddpg_baseline']
ddpg_hybrid = ['pendulum/ddpg_hybrid1', 'cartpole/ddpg_hybrid1_-60', 'mountaincar/ddpg_hybrid']
pilco_baseline = ['pendulum/pilco_baseline1', 'cartpole/pilco_baseline1', 'mountaincar/pilco_baseline1']
pilco_hybrid = ['pendulum/pilco_hybrid', 'cartpole/pilco_hybrid', 'mountaincar/pilco_hybrid']
linear_ctrl = ['pendulum/linear', 'cartpole/linear', 'mountaincar/linear']

env_names = ['Pendulum-v8', 'Cartpole-v8', 'Mountaincar-v8']
controllers = [ddpg_baseline, ddpg_hybrid, pilco_baseline]

# Append trajectories by env and controller
for i, env_name in enumerate(env_names):
    dict_of_dict_controller_name_scaling_trajectory = {} # type: Dict[str, Dict[float, List[Trajectory]]]
    for controller in controllers:
        dict_scaling_trajectory = {}
        for ratio in np.linspace(1.0, 2.0, 20):
            env = suite_gym.load(env_name, gym_kwargs={'top': True, 'scaling_ratio': ratio})
            py_env = tf_py_environment.TFPyEnvironment(env)
            path = '../controllers/' + controller[i]
            myEvaluator = Evaluator(eval_env=py_env, policy=None, plotter=None, model_path=path, eval_num_episodes=1)
            myEvaluator.load_policy()
            trajectory = myEvaluator(training_timesteps=0, save_model=False)
            dict_scaling_trajectory[ratio] = trajectory
        dict_of_dict_controller_name_scaling_trajectory[env_name] = dict_scaling_trajectory

    # myMetricsPlotter = ControlMetricsPlotter(py_envs[i])
    trajectories = []  # a list containing trajectories for the current env

    # myMetricsPlotter(trajectories)
