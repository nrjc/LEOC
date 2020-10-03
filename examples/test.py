import tensorflow as tf
import gin
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import numpy as np
from tf_agents.metrics import tf_metrics

from controller_utils import LQR
from dao.metrics import CompleteStateObservation

gin.enter_interactive_mode()
import gpflow

gpflow.config.set_default_float(tf.float32)

gin.parse_config_file('examples/config.gin')

# %%

from pilco.controllers import LinearController
import examples.envs

# %%

test_linear_control = False
SUBS = 3
bf = 30
maxiter = 50
max_action = 2.0
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])
m_init = np.reshape([-1.0, 0, 0.0], (1, 3))
S_init = np.diag([0.01, 0.05, 0.01])
T = 40
J = 4
N = 8
restarts = 2
model_save_dir = './'

# Set up objects and variables
env = suite_gym.load('Pendulum-v7')
A, B, C, Q = env.control()
W_matrix = LQR().get_W_matrix(A, B, Q, env='swing up')

# Wrap in tf env
env = TFPyEnvironment(env)

state_dim = 3
control_dim = 1

controller_linear = LinearController(state_dim=state_dim, control_dim=control_dim, W=W_matrix, max_action=max_action,
                                     time_step_spec=env.time_step_spec(), action_spec=env.action_spec())

# %%
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
state_obs = CompleteStateObservation()
observers = [num_episodes, env_steps, state_obs]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env, controller_linear, observers, num_episodes=2)

# %%
final_time_step, _ = driver.run(policy_state=())
# %%
num_episodes.result()
# %%
env_steps.result()
#%%
state_obs.result()