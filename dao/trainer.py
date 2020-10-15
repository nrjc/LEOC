from typing import List

import gin
import gpflow
from gpflow import set_trainable
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.trajectories.trajectory import Trajectory

from DDPG.ddpg import DDPG
from DDPG.utils import train_ddpg
from pilco.utils import train_pilco
from dao.metrics import CompleteStateObservation
import tensorflow as tf


class Trainer:
    def __init__(self, env: TFPyEnvironment, policy: tf.Module, saved_path=None):
        self.env = env
        self.policy = policy
        self.saved_path = saved_path
        self.saver = policy_saver.PolicySaver(self.policy, batch_size=None)

    def train(self) -> tf.Module:
        raise NotImplementedError

    def eval(self) -> List[Trajectory]:
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        state_obs = CompleteStateObservation()
        observers = [num_episodes, env_steps, state_obs]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(self.env, self.policy, observers, num_episodes=2)
        final_time_step, _ = driver.run(policy_state=())
        return state_obs.result()

    def save(self):
        self.saver.save(self.saved_path)

    def load(self):
        self.policy = tf.compat.v2.saved_model.load(self.saved_path)

    def visualise(self, steps=200):
        # To visualise and check on policy
        time_step = self.env.reset()
        for step in range(steps):
            self.env.render()
            if time_step.is_last():
                break
            action_step = self.policy.action(time_step)
            time_step = self.env.step(action_step.action)
        self.env.close()


@gin.configurable
class DDPGTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, ddpg: DDPG, saved_path: str):
        super().__init__(env, ddpg.agent.policy, saved_path)

    def train(self) -> DDPG:
        return train_ddpg()


@gin.configurable
class PILCOTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, controller: tf.Module, saved_path: str):
        super().__init__(env, controller, saved_path)

    def train(self) -> tf.Module:
        return train_pilco()
