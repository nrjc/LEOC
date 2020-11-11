import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from dao.trainer import Evaluator
from plotting.plotter import ControlMetricsPlotter

pendulum_py_env = suite_gym.load('Pendulum-v8')
pendulum_env = tf_py_environment.TFPyEnvironment(pendulum_py_env)
cartpole_py_env = suite_gym.load('Cartpole-v8')
cartpole_env = tf_py_environment.TFPyEnvironment(cartpole_py_env)
mountaincar_py_env = suite_gym.load('Mountaincar-v8')
mountaincar_env = tf_py_environment.TFPyEnvironment(mountaincar_py_env)
ddpg_baseline = ['pendulum/ddpg_baseline1', 'cartpole/ddpg_baseline2_-57', 'mountaincar/ddpg_baseline3_-124']
ddpg_hybrid = ['pendulum/ddpg_hybrid1', 'cartpole/ddpg_hybrid1_-60', 'mountaincar/ddpg_hybrid2']
pilco_baseline = ['pendulum/pilco_baseline3', 'cartpole/pilco_baseline1', 'mountaincar/pilco_baseline2']
pilco_hybrid = ['pendulum/pilco_hybrid3', 'cartpole/pilco_hybrid3', 'mountaincar/pilco_hybrid2']
linear_ctrl = ['pendulum/linear', 'cartpole/linear', 'mountaincar/linear']

py_envs = [pendulum_py_env, cartpole_py_env, mountaincar_py_env]
envs = [pendulum_env, cartpole_env, mountaincar_env]
controllers = [ddpg_baseline, ddpg_hybrid, pilco_baseline, pilco_hybrid]

all_trajectories = {}  # a dict of dict containing trajectories for all envs

for i in range(len(envs)):
    # Append trajectories by env and controller
    env_trajectories = {}  # a dict containing trajectories for the current env

    for controller in controllers:
        path = '../controllers/' + controller[i]
        myEvaluator = Evaluator(eval_env=envs[i], policy=None, plotter=None, model_path=path, eval_num_episodes=1)
        myEvaluator.load_policy()
        trajectory = myEvaluator(training_timesteps=0, save_model=False, step_input=-0.2)  # impulse_input: float = 0.0,
        env_trajectories[controller[i]] = trajectory

    env_name = py_envs[i].unwrapped.spec.id[:-3]
    all_trajectories[env_name] = env_trajectories

myMetricsPlotter = ControlMetricsPlotter(py_envs)
myMetricsPlotter(all_trajectories)
