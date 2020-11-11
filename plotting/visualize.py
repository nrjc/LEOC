import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tf_agents.environments import suite_gym, tf_py_environment

import dao.envs
from plotting.visualiser import Visualiser

# py_env = suite_gym.load('Pendulum-v7')
# model_path = '../controllers/pendulum/pilco_hybrid1'
# py_env = suite_gym.load('Cartpole-v7')
# model_path = '../controllers/cartpole/ddpg_hybrid1_-60'
py_env = suite_gym.load('Mountaincar-v7')
model_path = '../controllers/mountaincar/pilco_baseline2'
env = tf_py_environment.TFPyEnvironment(py_env)

visualizer = Visualiser(env, model_path)
visualizer(1000)
