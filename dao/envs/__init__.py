from gym import register

register(
    id='Pendulum-v7',
    entry_point='dao.envs.pendulum_env:SwingUpEnv',
    max_episode_steps=200,
)
register(
    id='Pendulum-v8',
    entry_point='dao.envs.pendulum_env:SwingUpEnv',
    max_episode_steps=200,
    kwargs={'top': True}
)
register(
    id='Cartpole-v7',
    entry_point='dao.envs.cartpole_env:CartPoleEnv',
    max_episode_steps=200,
)
register(
    id='Cartpole-v8',
    entry_point='dao.envs.cartpole_env:CartPoleEnv',
    max_episode_steps=200,
    kwargs={'top': True}
)
register(
    id='Mountaincar-v7',
    entry_point='dao.envs.mountain_car_env:Continuous_MountainCarEnv',
    max_episode_steps=200,
)
register(
    id='Mountaincar-v8',
    entry_point='dao.envs.mountain_car_env:Continuous_MountainCarEnv',
    max_episode_steps=200,
    kwargs={'top': True}
)
