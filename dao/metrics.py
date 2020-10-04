import tf_agents.trajectories
import gin
import tensorflow as tf
from tf_agents.utils import common


@gin.configurable(module='tf_agents')
class CompleteStateObservation:
    """Counts the number of steps taken in the environment."""

    def __init__(self, env_spec=(0, 3), dtype=tf.float32):
        self.dtype = dtype
        self.env_spec = env_spec
        self.dtype = dtype
        self._storage = []

    def __call__(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
        """Increase the number of environment_steps according to trajectory.

        Step count is not increased on trajectory.boundary() since that step
        is not part of any episode.

        Args:
          trajectory: A tf_agents.trajectory.Trajectory

        Returns:
          The arguments, for easy chaining.
        """
        # Zero out batch indices where a new episode is starting.
        self._storage.append(trajectory.observation)
        return trajectory

    def result(self):
        return self._storage

    @common.function
    def reset(self):
        self._storage = []
