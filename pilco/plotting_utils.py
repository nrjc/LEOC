import math
from typing import List
from matplotlib import pyplot as plt
import numpy as np


def plot_single_rollout_cycle(state_mean: List[np.ndarray], state_var: List[np.ndarray],
                              rollout: List[List[np.ndarray]], rollout_action: List[np.ndarray], internal_state_dim_num: int, action_dim_num: int,
                              time_steps: int, rollout_num: int):
    """

    Args:
        state_mean: A list (length T) of numpy array of dimension (N), denoting the predicted means at each time step.
        state_var: A list (length T) of numpy array of dimension (N), denoting the variance at each time step.
        rollout: A list (length S) of list (length T) of numpy array of dimension (N), denoting the observed parameter state.
        rollout_action: A list (length T) of numpy array of dimension (M), denoting the actions taken by rollout controller.
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
    # Calculate loss.
    fig, axs = plt.subplots(math.ceil(total_graphs / width), width)
    for i in range(total_graphs):
        cur_graph_pos_i, cur_graph_pos_j = i // width, i % width
        plot_states = i < internal_state_dim_num
        plot_actions = i >= internal_state_dim_num & i < (total_graphs - 1)
        plot_loss = ~plot_states & ~plot_actions
        cur_axis = axs[cur_graph_pos_i, cur_graph_pos_j]
        if plot_states:
            y = mean_states[:, i]
            yerr = var_states[:, i]
            cur_axis.errorbar(np.arange(time_steps), y, yerr)
            for s in range(rollout_num):
                cur_axis.plot(np.arange(time_steps), rollouts[s, :, i])
                # pass
        if plot_actions:
            # TODO: Plot actions
            continue
        if plot_loss:
            # TODO: Plot Loss
            continue
    fig.show()
