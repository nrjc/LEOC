from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tf_agents.agents import DdpgAgent
from tf_agents.agents.ddpg import actor_network, critic_network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver

tf.compat.v1.enable_v2_behavior()


class DDPG(object):

    def __init__(self, train_env, learning_rate, fc_layer_params=(100,)):
        self.train_env = train_env
        self.learning_rate = learning_rate
        self.fc_layer_params = fc_layer_params
        self.actor_network = actor_network.ActorNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.fc_layer_params,
            dropout_layer_params=None,
            conv_layer_params=None,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=None,
            last_kernel_initializer=None,
            name='DDPG_actor')
        self.critic_network = critic_network.CriticNetwork(
            (self.train_env.observation_spec(), self.train_env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            observation_dropout_layer_params=None,
            action_fc_layer_params=None,
            action_dropout_layer_params=None,
            joint_fc_layer_params=None,
            joint_dropout_layer_params=None,
            activation_fn=tf.nn.relu,
            output_activation_fn=None,
            kernel_initializer=None,
            last_kernel_initializer=None,
            name='DDPG_critic')
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step_counter = tf.Variable(0)
        self.agent = DdpgAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            ou_stddev=1.0,
            ou_damping=1.0,
            target_actor_network=None,
            target_critic_network=None,
            target_update_tau=1.0,
            target_update_period=1,
            dqda_clipping=None,
            td_errors_loss_fn=None,  # common.element_wise_squared_loss
            gamma=1.0,
            reward_scale_factor=1.0,
            gradient_clipping=None,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=self.train_step_counter,
            name='DDPG_agent')
        self.agent.initialize()
        self.policy_saver = PolicySaver(self.agent.collect_policy, batch_size=None)

    def compute_avg_return(self, eval_env, num_episodes=10):
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

    def __init__(self, ddpg, train_env, replay_buffer_max_length):
        self.agent = ddpg.agent
        self.train_env = train_env
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_max_length)

    def collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.buffer.add_batch(traj)

    def collect_data(self, environment, policy, steps=100):
        for _ in range(steps):
            self.collect_step(environment, policy)

    def get_dataset(self):
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = self.buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
        iterator = iter(dataset)
        return iterator


def train_agent(ddpg, replay_buffer, train_env, eval_env, num_iterations, num_eval_episodes, log_interval):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    ddpg.agent.train = common.function(ddpg.agent.train)

    # Reset the train step
    ddpg.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = ddpg.compute_avg_return(eval_env)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            replay_buffer.collect_step(train_env, ddpg.agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        iterator = replay_buffer.get_dataset()
        experience, unused_info = next(iterator)
        train_loss = ddpg.agent.train(experience).loss

        ddpg.policy_saver.save(f'policy_0')
        step = ddpg.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = ddpg.compute_avg_return(eval_env)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            # utils_plot(num_iterations, eval_interval, returns)
            ddpg.policy_saver.save(f'policy_{step // eval_interval}')

    print(f'Finished training for {num_iterations} iterations')


if __name__ == "__main__":
    num_iterations = 20000  # @param {type:"integer"}
    initial_collect_steps = 1000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}
    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    env_name = 'Pendulum-v0'
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    myDDPGagent = DDPG(train_env, learning_rate)
    myReplayBuffer = ReplayBuffer(myDDPGagent, train_env, replay_buffer_max_length)

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    myReplayBuffer.collect_data(train_env, random_policy, steps=100)
    train_agent(myDDPGagent,
                myReplayBuffer,
                train_env,
                eval_env,
                num_iterations,
                num_eval_episodes,
                log_interval)
