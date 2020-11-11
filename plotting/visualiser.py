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

    def __call__(self, steps: int = 100, impulse_input: float = 0.0, step_input: float = 0.0):
        assert impulse_input == 0.0 or step_input == 0.0, \
            '--- Error: Cannot do both impulse response and step response analysis simultaneously! ---'

        time_step = self.env.reset()
        for step in range(steps):
            self.env.render()
            if time_step.is_last():
                break
            action_step = self.policy.action(time_step)
            action, ratio = action_step.action, action_step.info

            if impulse_input != 0.0 and step == 0:
                action += impulse_input
            elif step_input != 0.0:
                action += step_input

            time_step = self.env.step(action)
        self.env.close()
