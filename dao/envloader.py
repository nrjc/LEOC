import gin
from tf_agents.environments import suite_gym
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from dao.controller_utils import LQR


@gin.configurable
def load_py_env(name: str) -> PyEnvironment:
    env1 = suite_gym.load(name)
    return env1


@gin.configurable
def load_weight_matrix(env: PyEnvironment):
    W_matrix = None
    try:
        A, B, C, Q = env.control()
        W_matrix = LQR().get_W_matrix(A, B, Q, env)
    except Exception:
        pass
    return W_matrix

@gin.configurable
def load_tf_py_env(env: PyEnvironment) -> TFPyEnvironment:
    return TFPyEnvironment(env)


def load_controller():
    pass


def TFPy2Gym(env: TFPyEnvironment) -> TFPyEnvironment:
    return env.pyenv.envs[0]._env.gym  # Dirty hacks all around