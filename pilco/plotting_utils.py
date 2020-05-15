from typing import List
from matplotlib import pyplot as plt
import numpy as np


def plot_single_rollout_cycle(state_mean: List[np.ndarray], state_var: List[np.ndarray],
                              rollout: List[List[np.ndarray]], rollout_action: List[np.ndarray], N: int, M: int,
                              T: int, S: int):
    """

    Args:
        state_mean: A list (length T) of numpy array of dimension (N), denoting the predicted means at each time step.
        state_var: A list (length T) of numpy array of dimension (N), denoting the variance at each time step.
        rollout: A list (length S) of list (length T) of numpy array of dimension (N), denoting the observed parameter state.
        rollout_action: A list (length T) of numpy array of dimension (M), denoting the actions taken by rollout controller.
        N: size of internal state dimensions
        M: size of action dimensions
        T: time steps
        S: represents the number of rollouts
    Returns:
        None

    """
    # TODO: Add titles
    width = 3
    total_graphs = N + M + 1
    mean_states = np.array(state_mean)
    var_states = np.array(state_var)
    rollouts = np.array(rollout)
    # Calculate loss.
    fig, axs = plt.subplots(total_graphs // width, width)
    for i in range(total_graphs):
        cur_graph_pos_i, cur_graph_pos_j = i // width, i % width
        plot_states = i < N
        plot_actions = i >= N & i < (total_graphs - 1)
        plot_loss = ~plot_states & ~plot_actions
        cur_axis = axs[cur_graph_pos_i, cur_graph_pos_j]
        if plot_states:
            y = mean_states[:, i]
            yerr = var_states[:, i]
            cur_axis.errorbar(np.arange(T), y, yerr)
            for s in range(S):
                cur_axis.plot(np.arange(T), rollouts[s, :, i])
        if plot_actions:
            # TODO: Plot actions
            continue
        if plot_loss:
            # TODO: Plot Loss
            continue
    fig.show()
