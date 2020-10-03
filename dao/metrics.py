import tf_agents.trajectories
from tf_agents.metrics import tf_metric
import gin
import tensorflow as tf
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common


@gin.configurable(module='tf_agents')
class CompleteStateObservation(tf_metric.TFStepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, name='EnvironmentSteps', prefix='Metrics', dtype=tf.int64):
    super(CompleteStateObservation, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self._all_trajectories = []

  def call(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
    """Increase the number of environment_steps according to trajectory.

    Step count is not increased on trajectory.boundary() since that step
    is not part of any episode.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    # The __call__ will execute this.
    self._all_trajectories.append(trajectory)
    return trajectory

  def result(self):
    return self._all_trajectories

  @common.function
  def reset(self):
    self._all_trajectories = []

