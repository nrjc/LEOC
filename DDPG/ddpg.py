r"""Train and Eval DDPG.
To run:
```bash
tensorboard --logdir $HOME/tmp/ddpg/gym/HalfCheetah-v2/ --port 2223 &
python tf_agents/agents/ddpg/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/ddpg/gym/HalfCheetah-v2/ \
  --num_iterations=2000000 \
  --alsologtostderr
```
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from absl import app, flags, logging

import tensorflow as tf
from tf_agents.agents import DdpgAgent
from tf_agents.agents.ddpg import actor_network, critic_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000, 'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
FLAGS = flags.FLAGS


class LinearController(tf.Module):
    def __init__(self, state_dim, control_dim, max_action=1.0, W=None, b=None, name=None):
        super().__init__(name=name)
        if W is None:
            self.w = tf.Variable(tf.random.normal([control_dim, state_dim]), trainable=True, name='W')
            self.b = tf.Variable(tf.zeros([control_dim]), trainable=False, name='b', dtype=tf.dtypes.float32)
        else:
            self.W = tf.Variable(W, trainable=False, name='W', dtype=tf.dtypes.float32)
            self.b = tf.Variable(tf.zeros([control_dim]), trainable=False, name='b')
        self.max_action = max_action

    def compute_action(self, x):
        '''
        Simple affine action:  u <- W(x_t) + b
        IN: state (x)
        OUT: action (u)
        '''
        x = tf.cast(x, dtype=tf.dtypes.float32, name='state')  # cast
        u = x @ tf.transpose(self.W) + self.b
        return u

    def randomize(self):
        mean = 0
        sigma = 1
        self.W.assign(mean + sigma * np.random.normal(size=self.W.shape))
        self.b.assign(mean + sigma * np.random.normal(size=self.b.shape))


class DDPG(object):

    def __init__(self, train_env, linear_controller=None):
        self.train_env = train_env
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.actor_network = actor_network.ActorNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=(400, 300),
            dropout_layer_params=None,
            conv_layer_params=None,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=None,
            last_kernel_initializer=None,
            name='DDPG_actor')
        self.critic_network = critic_network.CriticNetwork(
            (self.train_env.observation_spec(), self.train_env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=(400,),
            observation_dropout_layer_params=None,
            action_fc_layer_params=None,
            action_dropout_layer_params=None,
            joint_fc_layer_params=(300,),
            joint_dropout_layer_params=None,
            activation_fn=tf.nn.relu,
            output_activation_fn=None,
            kernel_initializer=None,
            last_kernel_initializer=None,
            name='DDPG_critic')
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.critic_learning_rate)
        self.train_step_counter = tf.Variable(0)
        self.agent = DdpgAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            ou_stddev=0.2,
            ou_damping=0.15,
            target_actor_network=None,
            target_critic_network=None,
            target_update_tau=0.05,
            target_update_period=5,
            dqda_clipping=None,
            td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
            gamma=0.995,
            reward_scale_factor=1.0,
            gradient_clipping=None,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=self.train_step_counter,
            name='DDPG_agent')
        self.agent.initialize()
        self.linear_controller = linear_controller
        self.policy_saver = policy_saver.PolicySaver(self.agent.collect_policy, batch_size=None)

    def compute_avg_return(self, eval_env, num_episodes=5):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = eval_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


class ReplayBuffer(object):

    def __init__(self, ddpg, train_env, replay_buffer_capacity, initial_collect_steps, collect_steps_per_iteration):
        self.agent = ddpg.agent
        self.train_env = train_env
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_capacity)
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()]
        self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.buffer.add_batch],
            num_steps=initial_collect_steps)
        self.collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.buffer.add_batch] + self.train_metrics,
            num_steps=collect_steps_per_iteration)

    def collect_step(self):
        time_step = self.train_env.current_time_step()
        action_step = self.agent.collect_policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.buffer.add_batch(traj)

    def collect_data(self, steps=100):
        for _ in range(steps):
            self.collect_step()

    def get_dataset(self, batch_size=64):
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = self.buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
        iterator = iter(dataset)
        return iterator


def train_agent(ddpg, replay_buffer, eval_env, num_iterations, initial_collect_steps=1000,
                collect_steps_per_iteration=1, num_eval_episodes=5, log_interval=200, eval_interval=1000):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    ddpg.agent.train = common.function(ddpg.agent.train)

    # Reset the train step
    ddpg.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = ddpg.compute_avg_return(eval_env)
    returns = [avg_return]

    # Collect some initial experience
    replay_buffer.collect_data(steps=initial_collect_steps)

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        replay_buffer.collect_data(steps=collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        iterator = replay_buffer.get_dataset()
        experience, _ = next(iterator)
        train_loss = ddpg.agent.train(experience).loss

        ddpg.policy_saver.save(f'policy_0')
        step = ddpg.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f'step = {step}: loss = {train_loss}')

        if step % eval_interval == 0:
            avg_return = ddpg.compute_avg_return(eval_env, num_episodes=num_eval_episodes)
            print(f'step = {step}: Average Return = {avg_return}')
            returns.append(avg_return)
            # utils_plot(num_iterations, eval_interval, returns)
            ddpg.policy_saver.save(f'policy_{step // eval_interval}')

    print(f'Finished training for {num_iterations} iterations')