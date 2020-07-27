from typing import Tuple, List
import numpy as np

import tensorflow as tf
import copy


def analyze_robustness(controller, env, function_boundaries: List[Tuple[float, float]], variable_names: List[str],
                       noise_values: np.ndarray):
    stable_percentage = np.empty_like(noise_values)
    for i, noise in enumerate(noise_values):
        stable_percentage[i] = percentage_stable(controller, env, function_boundaries, variable_names,
                                                 noise)
    return stable_percentage


def percentage_stable(controller, env, function_boundaries: List[Tuple[float, float]], variable_names: List[str],
                      noise: float, N=100) -> float:
    num_stable = 0
    for num in range(N):
        env_temp = copy.deepcopy(env)
        env_temp.mutate_with_noise(noise, variable_names)
        stable = is_stable(controller, env_temp, function_boundaries)
        if stable:
            num_stable += 1
    return num_stable / N


def is_stable(controller, env, function_boundaries: List[Tuple[float, float]]) -> bool:
    states = env.reset()
    stable = True
    for i in range(20):
        action = controller.compute_action(tf.reshape(tf.convert_to_tensor(states), (1, -1)),
                                           tf.zeros([env.observation_space_dim, env.observation_space_dim],
                                                    dtype=tf.dtypes.float64),
                                           squash=True)[0]
        action = action[0, :].numpy()
        states, _, _, _ = env.step(action)
        for dim, (lower, upper) in enumerate(function_boundaries):
            stable = stable and (states[dim] >= lower) and (states[dim] <= upper)
        if not stable:
            break
    return stable
