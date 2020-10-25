import copy
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
from dao.metrics import CompleteTrajectoryObservation
from dao.plotter import StatePlotter, Plotter

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
class Visualiser:
    """
    To visualise and check on policy
    """
    def __init__(self, env: TFPyEnvironment, policy: TFPolicy, saved_path: str = None):
        self.env = env
        self.policy = policy
        self.saved_path = saved_path

    def load(self):
        self.policy = tf.compat.v2.saved_model.load(self.saved_path)

    def __call__(self, steps=100):
        # self.env.pyenv.envs[0]._env.gym.up = True
        time_step = self.env.reset()
        for step in range(steps):
            self.env.render()
            if time_step.is_last():
                break
            action_step = self.policy.action(time_step)
            action, ratio = action_step.action, action_step.info
            time_step = self.env.step(action)
        self.env.close()


@gin.configurable
class Evaluator:
    def __init__(self, env: TFPyEnvironment, policy: TFPolicy, plotter: Plotter = None,
                 saved_path: str = None, eval_num_episodes: int = 1):
        self.env = env
        self.policy = policy
        self.saved_path = saved_path
        self.best_reward = -np.finfo(float_type).max
        self.saver = policy_saver.PolicySaver(self.policy, batch_size=None)
        self.eval_num_episodes = eval_num_episodes
        self.plotter = plotter

    def load(self):
        self.policy = tf.compat.v2.saved_model.load(self.saved_path)

    def save(self):
        self.saver.save(self.saved_path)

    def __call__(self, save_model: bool = False) -> List[Trajectory]:
        """
        Evaluate the trained policy, possibly saving model and plotting.
        Invoked after each eval_
        """
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        avg_reward = tf_metrics.AverageReturnMetric()
        trajectories = CompleteTrajectoryObservation()
        observers = [num_episodes, env_steps, avg_reward, trajectories]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env,
            self.policy,
            observers,
            num_episodes=self.eval_num_episodes)
        final_time_step, _ = driver.run(policy_state=())

        eval_reward = avg_reward.result()
        print(f'eval_reward from AverageReturnMetric = {eval_reward}')
        if save_model and eval_reward > self.best_reward:
            self.best_reward = eval_reward
            self.save()

        if self.plotter is not None:
            self.plotter(trajectories.result(), num_episodes=self.eval_num_episodes)

        return trajectories.result()


@gin.configurable
class DDPGTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, ddpg: DDPG, replay_buffer: ReplayBuffer,
                 num_iterations: int = 10000, eval_interval: int = 200):
        self.ddpg = ddpg
        super().__init__(env, self.ddpg.agent.policy)
        self.replay_buffer = replay_buffer
        self.num_iterations = num_iterations
        self.eval_interval = eval_interval
        self.evaluator = Evaluator(env, self.policy)

    def train(self, batch_size=64, initial_collect_steps=1000, collect_steps_per_rollout=1) -> DDPG:
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.ddpg.agent.train = common.function(self.ddpg.agent.train)

        # Reset the train step
        self.ddpg.train_step_counter.assign(0)
        returns = []
        # lambdas = [ddpg.actor_network.S.numpy()]

        # Collect some initial experience
        self.replay_buffer.collect_data(steps=initial_collect_steps)

        for iteration in range(self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.replay_buffer.collect_data(steps=collect_steps_per_rollout)

            # Sample a batch of data from the buffer and update the agent's network.
            iterator = self.replay_buffer.get_dataset(batch_size)
            experience, _ = next(iterator)

            # lambdas.append(ddpg.actor_network.S.numpy())

            # Eval, save model and plot
            if iteration % self.eval_interval == 0:
                print(f'--- Iteration {iteration} ---')
                save_model = iteration > int(self.num_iterations / 3 * 2)
                self.evaluator(save_model=save_model)

        print(f'--- Finished training for {self.num_iterations} iterations ---')
        return self.ddpg


@gin.configurable
class PILCOTrainer(Trainer):
    def __init__(self, env: TFPyEnvironment, controller: TFPolicy, weights: List[float], m_init: List[float],
                 S_init: List[float], num_rollouts: int = 10, eval_interval: int = 1):
        super().__init__(env, controller)
        self.weights = np.array(np.diag(weights), dtype=float_type)
        self.m_init = np.array(np.reshape(m_init, (1, -1)), dtype=float_type)
        self.S_init = np.array(np.diag(S_init), dtype=float_type)
        self.num_rollouts = num_rollouts
        self.eval_interval = eval_interval
        self.evaluator = Evaluator(env, controller)

    def train(self, initial_num_rollout: int = 3, timesteps: int = 40, subs: int = 3, max_iter_policy_train: int = 50,
              max_training_restarts: int = 2, max_policy_restarts: int = 2) \
            -> tf.Module:

        env = TFPy2Gym(self.env)
        target = env.target

        R = ExponentialReward(state_dim=self.env.observation_spec().shape[0], t=target, W=self.weights)

        # Initial random rollouts to generate a dataset
        X, Y, _, _ = rollout(env=env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=False,
                             verbose=False)
        for i in range(1, initial_num_rollout):
            X_, Y_, _, _ = rollout(env=env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=False,
                                   verbose=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        pilco = PILCO((X, Y), controller=self.policy, horizon=timesteps, reward=R, m_init=self.m_init,
                      S_init=self.S_init)

        # for numerical stability, we can set the likelihood variance parameters of the GP models
        for model in pilco.mgpr.models:
            model.likelihood.variance.assign(0.001)
            set_trainable(model.likelihood.variance, False)

        for rollouts in range(self.num_rollouts):
            print(f'--- Iteration {rollouts} ---')
            policy_restarts = 1 if rollouts > 3 else max_policy_restarts
            pilco.optimize_models(maxiter=max_iter_policy_train, restarts=max_training_restarts)
            pilco.optimize_policy(maxiter=max_iter_policy_train, restarts=policy_restarts)
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=timesteps, verbose=False, SUBS=subs)
            _, _, reward = pilco.predict(X_new[0, None, :-1], 0.001 * self.S_init, timesteps)

            # Update dataset
            X = np.vstack((X, X_new))
            Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))

            # Eval, save model and plot
            if rollouts % self.eval_interval == 0:
                save_model = rollouts > int(self.num_rollouts / 2)
                self.evaluator(save_model=save_model)

        print(f'--- Finished training for {self.num_rollouts} rollouts ---')
        return self.policy
