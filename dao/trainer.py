import os
import re
from os import listdir
from os.path import isfile, join
import pickle
import gin
import numpy as np
import tensorflow as tf
from typing import List, Union

from gpflow import set_trainable
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common

from DDPG.ddpg import DDPG, ReplayBuffer
from dao.envloader import TFPy2Gym
from dao.metrics import CompleteTrajectoryObservation, myDynamicEpisodeDriver
from plotting.plotter import AwardCurve, StatePlotter

from pilco.models import PILCO
from pilco.rewards import ExponentialReward
from pilco.utils import float_type, rollout


class Trainer:
    def __init__(self, env: TFPyEnvironment, policy: TFPolicy):
        self.env = env
        self.policy = policy

    def train(self, **kwargs) -> tf.Module:
        raise NotImplementedError


@gin.configurable
class Evaluator:
    def __init__(self, eval_env: TFPyEnvironment, policy: TFPolicy, plotter: StatePlotter = None,
                 load_model_path: str = None, eval_num_episodes: int = 1):
        self.env = eval_env
        self.policy = policy
        self.env_name = self.env.pyenv.envs[0].unwrapped.spec.id.split('-')[0]
        self.policy_name = self.policy.policy_name
        self.model_path = self._get_model_path()
        self.pickle_path = self._get_pickle_path()
        self.load_model_path = load_model_path

        self.best_reward = -np.finfo(float_type).max
        if self.policy is not None:
            self.saver = policy_saver.PolicySaver(self.policy, batch_size=None)
        self.eval_num_episodes = eval_num_episodes
        self.plotter = plotter
        self.eval_results = []
        self.eval_times = []

    def _get_idx(self, directory):
        def extract_number(f):
            s = re.findall(f'{self.policy_name}(\d+).*', f)
            idx = int(s[0]) if s else -1
            return idx

        # check directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
            return 0
        else:
            # find the appropriate index/name
            model_files = [f for f in listdir(directory)]
            try:
                return max([extract_number(f) for f in model_files]) + 1
            except ValueError:
                return 0

    def load_policy(self):
        if self.load_model_path:
            self.policy = tf.compat.v2.saved_model.load(self.load_model_path)
            print(f'Loaded policy {self.load_model_path}')

    def _get_model_path(self):
        model_dir = os.path.join('controllers', self.env_name)
        idx = self._get_idx(model_dir)
        foldername = self.policy_name + str(idx)
        return os.path.join(model_dir, foldername)

    def save_policy(self):
        self.saver.save(self.model_path)
        print(f'Policy saved at {self.model_path}')

    def _get_pickle_path(self):
        pickle_dir = os.path.join('pickle', self.env_name)
        idx = self._get_idx(pickle_dir)
        filename = self.policy_name + str(idx) + '.pickle'
        return os.path.join(pickle_dir, filename)

    def get_awardcurve(self):
        # save xy as pickle file
        xy = np.array([self.eval_times, self.eval_results])
        xy = AwardCurve(xy)
        return xy

    def save_pickle(self):
        xy = self.get_awardcurve()
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(xy, f)
        print(f'{self.policy_name} rewards saved to {self.pickle_path}')

    def __call__(self, training_time: Union[int, float], save_model: bool = False,
                 impulse_input: float = 0.0, step_input: float = 0.0) -> List[Trajectory]:
        """
        Invoked after each eval_interval during training phase.
        Each __call__ does 4 things:
            - Evaluate the trained policy.
            - Possibly saving model if returns are the best so far.
            - Does any plotting.
            - Keep the eval_reward for learning curve plotting.
        """
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        avg_reward = tf_metrics.AverageReturnMetric()
        trajectories = CompleteTrajectoryObservation()
        observers = [num_episodes, env_steps, avg_reward, trajectories]
        driver = myDynamicEpisodeDriver(
            env=self.env,
            policy=self.policy,
            impulse_input=impulse_input,
            step_input=step_input,
            observers=observers,
            num_episodes=self.eval_num_episodes)
        final_time_step, _ = driver.run(policy_state=())

        eval_reward = avg_reward.result()
        print(f'Evaluator eval_reward = {int(eval_reward)}, on average of {self.eval_num_episodes} episodes.')

        # save model
        if save_model and eval_reward > self.best_reward:
            self.best_reward = eval_reward
            self.save_policy()

        # plot graph
        if self.plotter is not None:
            self.plotter(trajectories.result(), num_episodes=self.eval_num_episodes)

        # append to learning curve
        self.eval_results.append(eval_reward.numpy())
        self.eval_times.append(training_time)

        return trajectories.result()


@gin.configurable
class DDPGTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, ddpg: DDPG, num_iterations: int = 10000, eval_interval: int = 200):
        self.ddpg = ddpg
        super().__init__(env, self.ddpg.agent.policy)
        self.tau = TFPy2Gym(self.env).tau
        self.replay_buffer = ReplayBuffer(self.ddpg)
        self.num_iterations = num_iterations
        self.eval_interval = eval_interval
        self.evaluator = Evaluator(env, self.policy)

    def train(self, batch_size=64, initial_collect_steps=1000, collect_steps_per_rollout=1) -> DDPG:
        self.ddpg.agent.train = common.function(self.ddpg.agent.train)
        # Reset the train step
        self.ddpg.train_step_counter.assign(0)
        # lambdas = [ddpg.actor_network.S.numpy()]
        # Collect some initial experience
        self.replay_buffer.collect_data(steps=initial_collect_steps)

        for iteration in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            self.replay_buffer.collect_data(steps=collect_steps_per_rollout)

            # Sample a batch of data from the buffer and update the agent's network.
            iterator = self.replay_buffer.get_dataset(batch_size)
            experience, _ = next(iterator)
            train_loss = self.ddpg.agent.train(experience).loss

            # lambdas.append(ddpg.actor_network.S.numpy())

            # Eval, save model and plot
            if iteration % self.eval_interval == 0:
                print(f'--- Iteration {iteration} ---')
                save_model = iteration > int(self.num_iterations / 3 * 2)
                self.evaluator(training_time=iteration * self.tau, save_model=save_model)

        # Update pickle file containing eval_results
        self.evaluator.save_pickle()

        print(f'--- Finished training for {self.num_iterations} iterations ---')
        return self.ddpg


@gin.configurable
class PILCOTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, controller: TFPolicy, weights: List[float], m_init: List[float],
                 S_init: List[float], num_rollouts: int = 10, eval_interval: int = 1):
        super().__init__(env, controller)
        self.state_dim = self.env.observation_spec().shape[0]
        self.env = TFPy2Gym(self.env)
        self.target = self.env.target
        self.tau = self.env.tau
        self.weights = np.array(np.diag(weights), dtype=float_type)
        self.m_init = np.array(np.reshape(m_init, (1, -1)), dtype=float_type)
        self.S_init = np.array(np.diag(S_init), dtype=float_type)
        self.num_rollouts = num_rollouts
        self.eval_interval = eval_interval
        self.evaluator = Evaluator(env, controller)

    def train(self, initial_num_rollout: int = 3, timesteps: int = 40, subs: int = 3, max_iter_policy_train: int = 50,
              max_training_restarts: int = 2, max_policy_restarts: int = 2) \
            -> tf.Module:

        R = ExponentialReward(state_dim=self.state_dim, t=self.target, W=self.weights)

        # Initial random rollouts to generate a dataset
        X, Y, _, _ = rollout(env=self.env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=False,
                             verbose=False)
        for i in range(1, initial_num_rollout):
            X_, Y_, _, _ = rollout(env=self.env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=False,
                                   verbose=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        pilco = PILCO((X, Y), controller=self.policy, horizon=timesteps, reward=R, m_init=self.m_init,
                      S_init=self.S_init)

        # Initial model evaluation
        self.evaluator(training_time=0, save_model=False)

        # for numerical stability, we can set the likelihood variance parameters of the GP models
        for model in pilco.mgpr.models:
            model.likelihood.variance.assign(0.001)
            set_trainable(model.likelihood.variance, False)

        for rollouts in range(self.num_rollouts):
            print(f'--- Iteration {rollouts} ---')
            policy_restarts = 1 if rollouts > 3 else max_policy_restarts
            pilco.optimize_models(maxiter=max_iter_policy_train, restarts=max_training_restarts)
            pilco.optimize_policy(maxiter=max_iter_policy_train, restarts=policy_restarts)
            X_new, Y_new, _, _ = rollout(self.env, pilco, timesteps=timesteps, verbose=False, SUBS=subs)
            _, _, reward = pilco.predict(X_new[0, None, :-1], 0.001 * self.S_init, timesteps)

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))

            # Eval, save model and plot
            if rollouts % self.eval_interval == 0:
                save_model = rollouts > int(self.num_rollouts / 2)
                training_timesteps = (rollouts + initial_num_rollout) * timesteps
                self.evaluator(training_time=training_timesteps * self.tau, save_model=save_model)

        # Update pickle file containing eval_results
        self.evaluator.save_pickle()

        print(f'--- Finished training for {self.num_rollouts} rollouts ---')
        return self.policy
