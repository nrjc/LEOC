import math
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from gym import logger


def plot_single_rollout_cycle(state_mean: List[np.ndarray], state_var: List[np.ndarray],
                              rollout: List[List[np.ndarray]], rollout_actions: List[np.ndarray],
                              all_rewards: List[np.ndarray], internal_state_dim_num: int, action_dim_num: int,
                              time_steps: int, rollout_num: int, env='swing up'):
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
        env: the experiment environment, string
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
        "--- Error: States dimensions do not match! ---"
    if rollout_actions is not None:
        actions = np.array(rollout_actions)
        assert actions.shape[1] == action_dim_num, "--- Error: Actions dimensions do not match! ---"
    # if rollout_reward is not None:
    #     reward = np.array(rollout_reward)

    if env == 'swing up':
        states_subtitles = ['cos(theta)', 'sin(theta)', 'theta dot']
        actions_subtitles = ['torque']
    elif env == 'cartpole':
        states_subtitles = ['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta dot']
        actions_subtitles = ['force']
    elif env == 'mountain car':
        states_subtitles = ['x', 'x_dot']
        actions_subtitles = ['force']
    else:
        logger.error("--- Error: plot_single_rollout_cycle() env incorrect! ---")
    assert len(states_subtitles) == internal_state_dim_num, "--- Error: Change states_subtitles! ---"
    assert len(actions_subtitles) == action_dim_num, "--- Error: Change actions_subtitles! ---"

    plt.style.use('seaborn-darkgrid')

    # Calculate reward.
    fig, axs = plt.subplots(math.ceil(total_graphs / width), width, constrained_layout=True)
    fig.suptitle(f'Rollout {rollout_num}', fontsize=16)
    for i in range(total_graphs):
        cur_graph_pos_i, cur_graph_pos_j = i // width, i % width
        plot_states = i < internal_state_dim_num
        plot_actions = i >= internal_state_dim_num and i < (total_graphs - 1) and rollout_actions is not None
        plot_reward = i == total_graphs - 1 and all_rewards is not None
        cur_axis = axs[cur_graph_pos_i, cur_graph_pos_j]

        # Plotting the state variables across time
        if plot_states:
            y = mean_states[:, i]
            yerr = var_states[:, i]
            cur_axis.plot(np.arange(time_steps), y, color='royalblue')
            cur_axis.fill_between(np.arange(time_steps), y-yerr, y+yerr, alpha=0.5, facecolor='royalblue')
            early_termination_time_steps = rollouts.shape[1]
            cur_axis.plot(np.arange(early_termination_time_steps), rollouts[0, :, i], color='darkorange')
            cur_axis.set_title(f'State: {states_subtitles[i]}')

        # Plotting the actions across time
        if plot_actions:
            # Plot one of the M subplots for actions
            j = i - internal_state_dim_num
            cur_axis.plot(np.arange(time_steps), actions[:, j])
            cur_axis.set_title(f'Action: {actions_subtitles[j]}')

        # Plotting the rewards across time
        if plot_reward:
            cur_axis.plot(np.arange(rollout_num + 1), all_rewards)
            cur_axis.set_title(f'Reward')
    fig.show()


def plot_interaction_barchart(y_extended, y_pilco):
    # width of the bars
    barWidth = 0.2

    # The x position of bars
    r1 = np.arange(len(y_pilco))
    r2 = [x + barWidth for x in r1]

    plt.style.use('seaborn-darkgrid')

    # Create the bars
    plt.bar(r1, y_pilco, width=barWidth, color='royalblue', capsize=15, label='PILCO')
    plt.bar(r2, y_extended, width=barWidth, color='orange', capsize=15, label='Extended RBF controller')

    # general layout
    plt.xticks([r + barWidth/2 for r in range(len(y_extended))], ['Swing-up pendulum', 'Cartpole', 'Mountain car'])
    plt.ylabel('Interaction time')
    plt.legend()

    # Show graphic
    plt.show()


if __name__ == '__main__':

    # Choose the height of the bars
    y_extended = [30, 9.6, 30]
    y_pilco = [30, 17.5, 30]

    plot_interaction_barchart(y_extended, y_pilco)