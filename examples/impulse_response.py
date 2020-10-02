import argparse

import gym
import tensorflow as tf
import tf_agents.policies.policy_loader
import numpy as np
from tf_agents.environments import tf_py_environment, suite_gym

import examples.envs
from DDPG.ddpg import LinearControllerLayer
from controller_utils import LQR

environment_names = ['Pendulum-v7', 'Cartpole-v7', 'Mountaincar-v7']
parser = argparse.ArgumentParser(description='Generate Impulse response outputs.')
parser.add_argument('modelname', choices=environment_names)
parser.add_argument('--path', type=str)
parser.add_argument('--type', choices=['DDPG', 'PILCO'])
args = parser.parse_args()

modelname = args.modelname
env = suite_gym.load(modelname)
A, B, C, Q = env.unwrapped.control()
W_matrix = LQR().get_W_matrix(A, B, Q, env='swing up')
# linear_controller = LinearControllerLayer(env.unwrapped.observation_space.shape[0],
#                                           env.unwrapped.action_space.shape[0], W_matrix)
controller = tf_agents.policies.policy_loader.load(args.path)
env = tf_py_environment.TFPyEnvironment(env)
starting_spec = env.reset()
for i in range(500):
    env.render()
    action = controller.action(starting_spec)
    states, _, _, _ = env.step(action)
    print(f'Step: {i}, action: {action}')
