from typing import List, Union

import gin
import gpflow
import gym
from gpflow import set_trainable
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.trajectory import Trajectory
import numpy as np

from DDPG.ddpg import train_ddpg, DDPG
from pilco.utils import train_pilco
from dao.metrics import CompleteStateObservation
import tensorflow as tf


class Trainer:
    def __init__(self, env: TFPyEnvironment, model: tf.Module, saved_path=None):
        self.env = env
        self.model = model
        self.saved_path = saved_path

    def train(self) -> tf.Module:
        raise NotImplementedError

    def eval(self) -> List[Trajectory]:
        raise NotImplementedError

    def save(self):
        tf.saved_model.save(self.model, self.saved_path)

    def load(self):
        self.model = tf.saved_model.load(self.saved_path)


@gin.configurable
class DDPGTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, ddpg: DDPG, saved_path: str):
        super().__init__(env, ddpg.agent, saved_path)

    def train(self) -> DDPG:
        return train_ddpg()

    def eval(self) -> List[Trajectory]:
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        state_obs = CompleteStateObservation()
        observers = [num_episodes, env_steps, state_obs]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env, self.ddpg_agent.collect_policy, observers, num_episodes=2)
        final_time_step, _ = driver.run(policy_state=())
        return state_obs.result()


@gin.configurable
class PILCOTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, controller: tf.Module, saved_path: str):
        super().__init__(env, controller, saved_path)

    def train(self) -> tf.Module:
        train_pilco()

    def eval(self) -> List[Trajectory]:
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        state_obs = CompleteStateObservation()
        observers = [num_episodes, env_steps, state_obs]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env, self.controller, observers, num_episodes=2)
        final_time_step, _ = driver.run(policy_state=())
        return state_obs.result()
