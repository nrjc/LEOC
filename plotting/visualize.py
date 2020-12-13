import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from plotting.visualiser import Visualiser

# py_env = suite_gym.load('Pendulum-v7')
# model_path = 'controllers/pendulum/pilco_hybrid0'
py_env = suite_gym.load('Cartpole-v8')
model_path = 'controllers/cartpole/ddpg_baseline1'
# py_env = suite_gym.load('Mountaincar-v7')
# model_path = 'controllers/mountaincar/ddpg_baseline0'
env = tf_py_environment.TFPyEnvironment(py_env)

visualizer = Visualiser(env, model_path)
visualizer(400)  #, step_input=-0.2)