from typing import List, Union

import gin
import gym
from gpflow import set_trainable
from tf_agents.agents import DdpgAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.trajectory import Trajectory
import numpy as np

from DDPG.ddpg import train_agent, DDPG
from dao.metrics import CompleteStateObservation
from pilco.models import PILCO
from pilco.rewards import ExponentialReward
from utils import rollout
import tensorflow as tf

class Trainer:
    def __init__(self, env: TFPyEnvironment):
        self.env = env

    def train(self) -> tf.Module:
        raise NotImplementedError

    def eval(self) -> List[Trajectory]:
        raise NotImplementedError


@gin.configurable
class DDPGTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, ddpg: DDPG):
        super().__init__(env)
        self.ddpg_agent = ddpg.agent

    def train(self) -> DDPG:
        return train_agent()

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
    def __init__(self, env: TFPyEnvironment, controller: Union[PyPolicy, tf.Module], target: List[float], weights: List[float],
                 m_init: List[float], S_init: List[float],
                 initial_num_rollout: int = 3, training_rollout_total_num: int = 15, timesteps: int = 40, subs: int = 3,
                 max_iter_policy_train: int = 50, max_training_restarts: int = 2):
        super().__init__(env)
        self.controller = controller
        self.target = np.array(target)
        self.weights = np.diag(weights)
        self.initial_num_rollout = initial_num_rollout
        self.training_rollout_total_num = training_rollout_total_num
        self.subs = subs
        self.timesteps = timesteps
        self.max_iter_policy_train = max_iter_policy_train
        self.max_training_restarts = max_training_restarts
        self.m_init = np.reshape(m_init, (1, -1))
        self.S_init = np.diag(S_init)
        self.env = env.pyenv

    def train(self) -> tf.Module:
        R = ExponentialReward(state_dim=self.env.observation_spec().shape[0], t=self.target, W=self.weights)
        env = self.env.envs[0]._env.gym # Dirty hacks all around
        # Initial random rollouts to generate a dataset
        X, Y, _, _ = rollout(env=env, pilco=None, timesteps=self.timesteps, random=True, SUBS=self.subs, render=True,
                             verbose=False)
        for i in range(1, self.initial_num_rollout):
            X_, Y_, _, _ = rollout(env=env, pilco=None, timesteps=self.timesteps, random=True, SUBS=self.subs,
                                   render=True,
                                   verbose=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        pilco = PILCO((X, Y), controller=self.controller, horizon=self.timesteps, reward=R, m_init=self.m_init,
                      S_init=self.S_init)

        # for numerical stability, we can set the likelihood variance parameters of the GP models
        for model in pilco.mgpr.models:
            model.likelihood.variance.assign(0.001)
            set_trainable(model.likelihood.variance, False)

        for rollouts in range(self.training_rollout_total_num):
            print("**** ITERATION no", rollouts, " ****")
            policy_restarts = 1 if rollouts > 3 else 2
            pilco.optimize_models(maxiter=self.max_iter_policy_train, restarts=self.max_training_restarts)
            pilco.optimize_policy(maxiter=self.max_iter_policy_train, restarts=policy_restarts)
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=self.timesteps, verbose=False, SUBS=self.subs)

            # Since we had decide on the various parameters of the reward function
            # we might want to verify that it behaves as expected by inspection

            _, _, reward, intermediary_dict = pilco.predict_and_obtain_intermediates(X_new[0, None, :-1],
                                                                                     0.001 * self.S_init,
                                                                                     self.timesteps)

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
        return self.controller

    def eval(self) -> List[Trajectory]:
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        state_obs = CompleteStateObservation()
        observers = [num_episodes, env_steps, state_obs]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env, self.controller, observers, num_episodes=2)
        final_time_step, _ = driver.run(policy_state=())
        return state_obs.result()
