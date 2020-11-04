r"""Train and Eval DDPG.
To run:
```bash
tensorboard --logdir $HOME/tmp/ddpg/gym/HalfCheetah-v2/ --port 2223 &
python tf_agents/agents/ddpg/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/ddpg/gym/HalfCheetah-v2/ \
  --num_iterations=100 \
  --alsologtostderr
```
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import gin
from gpflow import config
from typing import Optional

import tensorflow as tf
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import policy_step
from tf_agents.agents import DdpgAgent, tf_agent
from tf_agents.agents.ddpg import actor_network, critic_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import actor_policy, ou_noise_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from dao.controller_utils import scale_to_spec
from dao.envloader import TFPy2Gym

float_type = config.default_float()

tf.compat.v1.enable_v2_behavior()


@gin.configurable
class LinearControllerLayer(tf.Module):
    def __init__(self, env: TFPyEnvironment, W=None, trainable=False):
        super(LinearControllerLayer, self).__init__()
        state_dim = env.observation_spec().shape[0]
        control_dim = env.action_spec().shape[0]
        if W is None:
            self.W = tf.Variable(tf.random.normal([control_dim, state_dim]), dtype=float_type, trainable=True, name='W')
        else:
            self.W = tf.Variable(W, dtype=float_type, trainable=False, name='W')
        self.b = tf.Variable(tf.zeros([control_dim], dtype=float_type), dtype=float_type, trainable=False, name='b')

    @tf.function
    def __call__(self, x):
        '''
        Simple affine action:  y <- W(x_t) + b
        IN: state (x)
        OUT: action (y)
        '''
        x = tf.cast(x, dtype=float_type, name='state')
        y = x @ tf.transpose(self.W) + self.b
        return y


class MyActorNetwork(actor_network.ActorNetwork):
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 linear_controller=None,
                 controller_location=None,
                 S=None,
                 **kwargs):
        super().__init__(input_tensor_spec=input_tensor_spec,
                         output_tensor_spec=output_tensor_spec,
                         **kwargs)
        self.linear_controller = linear_controller
        if controller_location is None:
            controller_location = tf.zeros(shape=input_tensor_spec.shape, dtype=float_type)
        if S is None:
            S = [1. for i in range(input_tensor_spec.shape[0])]
        self.a = tf.Variable(initial_value=controller_location, dtype=float_type, trainable=False)
        # self.S = tf.Variable(tf.ones(shape=input_tensor_spec.shape, dtype=float_type),
        #                      constraint=lambda x: tf.clip_by_value(x, 0, np.infty), trainable=True)
        self.S = tf.Variable(S, dtype=float_type, trainable=True)
        self.r = tf.Variable(tf.zeros((1,), dtype=float_type), trainable=False, name='ratio')
        # self.ratio = tf.Variable(tf.zeros(shape=self.S.shape, dtype=float_type), name='ratio')

    def compute_ratio(self, x):
        '''
        Compute the ratio of the linear controller
        '''
        if self.linear_controller:
            S = self.S
            a = self.a
            d = (x - a) @ tf.linalg.diag(S) @ tf.transpose(x - a)
            d_diag = tf.linalg.diag_part(d)
            ratio = 1 / tf.pow(d_diag + 1, 2)
            ratio = tf.expand_dims(ratio, axis=-1)
            self.r = ratio
        return self.r

    def call(self, observations, step_type=(), network_state=(), training=False):
        # output_actions, network_state = super().call(observations, step_type=(), network_state=(), training=False)
        del step_type  # unused.

        r = self.compute_ratio(observations)  # calculate ratio of linear controller
        if self.linear_controller:
            g_actions = self.linear_controller(observations)  # calculate linear action
        else:
            g_actions = 0.0
            g_actions = tf.cast(g_actions, float_type)  # convert to float64
        observations = tf.nest.flatten(observations)
        output = observations[0]
        for layer in self._mlp_layers:
            output = layer(output, training=training)
        h_actions = tf.cast(output, float_type)  # non-linear action
        actions = r * g_actions + (1 - r) * h_actions  # combined action
        actions = scale_to_spec(actions, self._single_action_spec)  # squash actions into action_spec bounds
        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, [actions])

        return output_actions, network_state


class MyActorPolicy(actor_policy.ActorPolicy):
    def __init__(self, **kwargs):
        info_spec = BoundedArraySpec((1,), float_type, minimum=0.0, maximum=1.0)
        super().__init__(info_spec=info_spec, **kwargs)

    def _distribution(self, time_step, policy_state):
        policy_step_super = super()._distribution(time_step, policy_state)
        distributions = policy_step_super.action
        policy_state = policy_step_super.state
        ratio = tf.cast(self._actor_network.r, float_type)
        return policy_step.PolicyStep(distributions, policy_state, ratio)


class MyDdpgAgent(DdpgAgent):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[tf.Variable] = None,
                 **kwargs):
        super().__init__(time_step_spec,
                         action_spec,
                         debug_summaries=debug_summaries,
                         summarize_grads_and_vars=summarize_grads_and_vars,
                         train_step_counter=train_step_counter,
                         **kwargs)
        policy = MyActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)
        collect_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)
        collect_policy = ou_noise_policy.OUNoisePolicy(
            collect_policy,
            ou_stddev=self._ou_stddev,
            ou_damping=self._ou_damping,
            clip=True)

        super(DdpgAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=2 if not self._actor_network.state_spec else None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)


@gin.configurable
class DDPG(tf.Module):
    def __init__(self, env: TFPyEnvironment, linear_controller=None, S=None, name='DDPG_agent', ):
        super().__init__(name=name)
        self.env = env
        self.actor_learning_rate = 1e-3
        self.critic_learning_rate = 1e-3
        self.actor_network = MyActorNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(400, 300),
            dropout_layer_params=None,
            conv_layer_params=None,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=None,
            last_kernel_initializer=None,
            linear_controller=linear_controller,
            controller_location=TFPy2Gym(self.env).target,
            S=S,
            name='DDPG_actor')
        self.critic_network = critic_network.CriticNetwork(
            (self.env.observation_spec(), self.env.action_spec()),
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
        self.agent = MyDdpgAgent(
            time_step_spec=self.env.time_step_spec(),
            action_spec=self.env.action_spec(),
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            ou_stddev=.5,
            ou_damping=.2,
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


@gin.configurable
class ReplayBuffer(object):
    def __init__(self, ddpg, replay_buffer_capacity=100000, initial_collect_steps=1000,
                 collect_steps_per_iteration=1):
        self.agent = ddpg.agent
        self.env = ddpg.env
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=replay_buffer_capacity)
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()]
        self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.env,
            self.agent.collect_policy,
            observers=[self.buffer.add_batch],
            num_steps=initial_collect_steps)
        self.collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.env,
            self.agent.collect_policy,
            observers=[self.buffer.add_batch] + self.train_metrics,
            num_steps=collect_steps_per_iteration)

    def collect_step(self):
        time_step = self.env.current_time_step()
        action_step = self.agent.collect_policy.action(time_step)
        next_time_step = self.env.step(action_step.action)
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
