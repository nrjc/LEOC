import tf_agents.trajectories
import gin
import tensorflow as tf
from tf_agents.utils import common


class CompleteObservation:
    """ABC for other observers."""

    def __init__(self):
        self._storage = []

    def __call__(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
        """Append the observed metric according to trajectory.

        Args:
          trajectory: A tf_agents.trajectory.Trajectory

        Returns:
          The arguments, for easy chaining.
        """
        raise NotImplementedError

    def result(self):
        return self._storage

    @common.function
    def reset(self):
        self._storage = []


@gin.configurable(module='tf_agents')
class CompleteTrajectoryObservation(CompleteObservation):
    """Keeps track of the entire trajectory."""

    def __init__(self):
        super(CompleteTrajectoryObservation, self).__init__()

    def __call__(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
        self._storage.append(trajectory)
        return trajectory


@gin.configurable(module='tf_agents')
class CompleteStateObservation(CompleteObservation):
    """Keeps track of the states from the trajectories."""

    def __init__(self):
        super(CompleteStateObservation, self).__init__()

    def __call__(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
        self._storage.append(trajectory.observation)
        return trajectory


@gin.configurable(module='tf_agents')
class CompleteActionInfoObservation(CompleteObservation):
    """Keeps track of the actions from the trajectories."""

    def __init__(self):
        super(CompleteActionInfoObservation, self).__init__()

    def __call__(self, trajectory: tf_agents.trajectories.trajectory.Trajectory):
        self._storage.append(trajectory.policy_info)
        return trajectory
