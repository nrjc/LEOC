import numpy as np
import control
import tensorflow as tf
import tensorflow_probability as tfp
from gym import logger
from tf_agents.environments.py_environment import PyEnvironment
from gpflow import config

float_type = config.default_float()


class LQR:
    def get_W_matrix(self, A, B, Q, env: PyEnvironment):
        # Compute gain matrix by solving Algebraic Riccati Equation
        # Reference https://www.mathworks.com/help/control/ref/lqr.html
        R = 1
        K, _, _ = control.lqr(A, B, Q, R)
        K = self.get_k_prime(K, env)
        return np.array(K)

    def get_k_prime(self, K, env: PyEnvironment):
        # Convert K to ndarray
        K = K.A
        # The internal states of gym envs are different from the internal states of theory, need to reorder the gains
        if env.unwrapped.spec.id == 'Pendulum-v7':
            # K := [theta, thetadot]
            # 'Pendulum-v0' gym env states = [cos(theta), sin(theta), thetadot]
            K_prime = [[0, K[0][0], K[0][1]]]

        elif env.unwrapped.spec.id == 'Cartpole-v7':
            # K := [x, x_dot, theta, theta_dot]
            # Cartpole env states = [x, x_dot, np.cos(theta), np.sin(theta), theta_dot]
            K_prime = [[K[0][0], K[0][1], 0, K[0][2], K[0][3]]]

        elif env.unwrapped.spec.id == 'Mountaincar-v7':
            # K := [position, velocity]
            # Mountain car gym env states = [position, velocity]
            K_prime = K

        else:
            logger.error("--- Error: LQR.get_k_prime() env incorrect! ---")
        return K_prime


def calculate_ratio(x, a, S):
    d = (x - a) @ np.diag(S) @ (x - a).transpose()
    ratio = 1 / np.power(d + 1, 2)
    return ratio


def to_distribution(action_or_distribution):
    if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
    return action_or_distribution


def spec_means_and_magnitudes(action_spec):
    """Get the center and magnitude of the ranges in action spec."""
    action_means = tf.nest.map_structure(
        lambda spec: (spec.maximum + spec.minimum) / 2.0, action_spec)
    action_magnitudes = tf.nest.map_structure(
        lambda spec: (spec.maximum - spec.minimum) / 2.0, action_spec)
    return action_means, action_magnitudes


def scale_to_spec(tensor, spec):
    """Shapes and scales a batch into the given spec bounds.

    Args:
      tensor: A [batch x n] tensor with values in the range of [-1, 1].
      spec: (BoundedTensorSpec) to use for scaling the action.

    Returns:
      A batch scaled the given spec bounds.
    """
    tensor = tf.reshape(tensor, [-1] + spec.shape.as_list())

    # Scale the tensor.
    means, magnitudes = spec_means_and_magnitudes(spec)
    tensor = means + magnitudes * tensor

    # Set type.
    return tf.cast(tensor, spec.dtype)
