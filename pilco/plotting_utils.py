import math
from typing import List
from matplotlib import pyplot as plt
import numpy as np


def plot_single_rollout_cycle(state_mean: List[np.ndarray], state_var: List[np.ndarray],
                              rollout: List[List[np.ndarray]], rollout_actions: List[np.ndarray],
                              all_rewards: List[np.ndarray],
                              internal_state_dim_num: int, action_dim_num: int,
                              time_steps: int, rollout_num: int):
    """

    Args:
        state_mean: T x N. A list (length T) of numpy array of dimension (N), denoting the predicted means at each time step.
        state_var: T x N. A list (length T) of numpy array of dimension (N), denoting the variance at each time step.
        rollout: S x T x N. A list (length S) of list (length T) of numpy array of dimension (N), denoting the observed parameter state.
        rollout_action: T x M. A list (length T) of numpy array of dimension (M), denoting the actions taken by rollout controller.
        all_rewards: 1 x up to N. A list (length up to N) of tensors, denoting the reward at the end of all previous rollouts.
        internal_state_dim_num: size of internal state dimensions
        action_dim_num: size of action dimensions
        time_steps: time steps
        rollout_num: represents the number of rollouts
    Returns:
        None

    """
    # TODO: Add titles
    width = 3
    total_graphs = internal_state_dim_num + action_dim_num + 1
    mean_states = np.array(state_mean)
    var_states = np.array(state_var)
    rollouts = np.array(rollout)
    assert mean_states.shape[1] == internal_state_dim_num and var_states.shape[1] == internal_state_dim_num, \
        "Error: States dimensions do not match!"
    if rollout_actions is not None:
        actions = np.array(rollout_actions)
        assert actions.shape[1] == action_dim_num, "Error: Actions dimensions do not match!"
    # if rollout_reward is not None:
    #     reward = np.array(rollout_reward)

    states_subtitles = ['cos(theta)', 'sin(theta)', 'theta dot']
    assert len(states_subtitles) == internal_state_dim_num, "Error: Change states_subtitles!"
    actions_subtitles = ['torque']
    assert len(actions_subtitles) == action_dim_num, "Error: Change actions_subtitles!"

    # Calculate reward.
    fig, axs = plt.subplots(math.ceil(total_graphs / width), width)
    for i in range(total_graphs):
        cur_graph_pos_i, cur_graph_pos_j = i // width, i % width
        plot_states = i < internal_state_dim_num
        plot_actions = i >= internal_state_dim_num and i < (total_graphs - 1) and rollout_actions is not None
        plot_reward = i == total_graphs - 1 and all_rewards is not None
        cur_axis = axs[cur_graph_pos_i, cur_graph_pos_j]
        if plot_states:
            y = mean_states[:, i]
            yerr = var_states[:, i]
            cur_axis.errorbar(np.arange(time_steps), y, yerr)
            # for s in range(rollout_num):
            #     cur_axis.plot(np.arange(time_steps), rollouts[s, :, i])
            cur_axis.plot(np.arange(time_steps), rollouts[0, :, i])
            cur_axis.set_title(f'State: {states_subtitles[i]}')
        if plot_actions:
            # Plot one of the M subplots for actions
            j = i - internal_state_dim_num
            cur_axis.plot(np.arange(time_steps), actions[:, j])
            cur_axis.set_title(f'Action: {actions_subtitles[j]}')
        if plot_reward:
            cur_axis.plot(np.arange(rollout_num), all_rewards)
            cur_axis.set_title(f'Reward')
    fig.show()
