from examples.envs.pendulum_env import SwingUpEnv
from examples.envs.cartpole_env import CartPoleEnv
from examples.envs.mountain_car_env import Continuous_MountainCarEnv
from gym import register

register(
    id='Pendulum-v7',
    entry_point='examples.envs.pendulum_env:SwingUpEnv',
    max_episode_steps=200,
)

register(
    id='Cartpole-v7',
    entry_point='examples.envs.cartpole_env:CartPoleEnv',
    max_episode_steps=200,
)

register(
    id='Mountaincar-v7',
    entry_point='examples.envs.mountain_car_env:Continuous_MountainCarEnv',
    max_episode_steps=200,
)