import tf_agents.trajectories
import gin
import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.trajectories import trajectory, policy_step
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


class myDynamicEpisodeDriver(DynamicEpisodeDriver):

    def __init__(self, env, policy, impulse_input: float = 0.0, step_input: float = 0.0, **kwargs):
        super().__init__(env, policy, **kwargs)

        assert impulse_input == 0.0 or step_input == 0.0, \
            '--- Error: Cannot do both impulse response and step response analysis simultaneously! ---'
        self.impulse_input = impulse_input
        self.step_input = step_input

    def _loop_body_fn(self):
        """Returns a function with the driver's loop body ops."""

        def loop_body(counter, time_step, policy_state):
            """Runs a step in environment.

            While loop will call multiple times.

            Args:
              counter: Episode counters per batch index. Shape [batch_size].
              time_step: TimeStep tuple with elements shape [batch_size, ...].
              policy_state: Poicy state tensor shape [batch_size, policy_state_dim].
                Pass empty tuple for non-recurrent policies.
            Returns:
              loop_vars for next iteration of tf.while_loop.
            """
            action_step = self.policy.action(time_step, policy_state)

            action = action_step.action
            if self.impulse_input != 0.0 and time_step.step_type == 0:
                action += self.impulse_input
            elif self.step_input != 0.0:
                action += self.step_input
            policy_state = action_step.state
            info = action_step.info
            action_step = policy_step.PolicyStep(action, policy_state, info)

            with tf.control_dependencies(tf.nest.flatten([time_step])):
                next_time_step = self.env.step(action_step.action)

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            observer_ops = [observer(traj) for observer in self._observers]
            transition_observer_ops = [
                observer((time_step, action_step, next_time_step))
                for observer in self._transition_observers
            ]
            with tf.control_dependencies(
                    [tf.group(observer_ops + transition_observer_ops)]):
                time_step, next_time_step, policy_state = tf.nest.map_structure(
                    tf.identity, (time_step, next_time_step, policy_state))

            # While loop counter is only incremented for episode reset episodes.
            counter += tf.cast(traj.is_boundary(), dtype=tf.int32)

            return [counter, next_time_step, policy_state]

        return loop_body