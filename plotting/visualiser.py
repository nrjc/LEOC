import gin
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment


@gin.configurable
class Visualiser:
    """
    To visualise and check on policy
    """

    def __init__(self, env: TFPyEnvironment, model_path: str = None):
        self.env = env
        self.model_path = model_path
        if model_path:
            self.load()

    def load(self):
        self.policy = tf.compat.v2.saved_model.load(self.model_path)

    def __call__(self, steps=100):
        time_step = self.env.reset()
        for step in range(steps):
            self.env.render()
            if time_step.is_last():
                break
            action_step = self.policy.action(time_step)
            action, ratio = action_step.action, action_step.info
            time_step = self.env.step(action)
        self.env.close()
