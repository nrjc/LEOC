import re
from typing import List, Tuple, Dict, Union
import gin
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.trajectory import Trajectory

plt.style.use('seaborn-darkgrid')


class TrajectoryPath(object):
    """
    This class handles related methods for a single epoch from a list of trajectories generated by eval
    """

    def __init__(self):
        self.observations = []
        self.rewards = []
        self.actions = []
        self.ratio = []
        self.theta = []
        self.timesteps = 0

        # initialise some dicts used to easy plotting
        self.color_dict, self.label_dict = RegexDict(), RegexDict()
        colors = ['dodgerblue', 'mediumblue', 'orange', 'firebrick', 'green']
        labels = ['DDPG_baseline', 'DDPG_hybrid', 'PILCO_baseline', 'PILCO_hybrid', 'linear/DDPG_hybrid/PILCO_hybrid']
        controller_regex = ['.*ddpg_baseline.*', '.*ddpg_hybrid.*', '.*pilco_baseline.*', '.*pilco_hybrid.*',
                            '.*linear.*']
        for i in range(len(controller_regex)):
            self.color_dict[controller_regex[i]] = colors[i]
            self.label_dict[controller_regex[i]] = labels[i]

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> None:
        raise NotImplementedError

    def _init_env(self, env_name: str):
        self.env_name = env_name
        # Depending on the env, sin(theta) or state of interest is located at different obs_idx
        if self.env_name == 'Pendulum':
            self.asin = True
            self.acos = False
            self.obs_idx = 1
        elif self.env_name == 'Cartpole':
            self.asin = True
            self.acos = False
            self.obs_idx = 3
        elif self.env_name == 'Mountaincar':
            self.asin = False
            self.acos = False
            self.obs_idx = 0
        else:
            raise Exception(f'--- Error: Wrong env in {self} ---')

    def _helper(self, trajectories: List[Trajectory], num_episodes: int = 1) -> None:
        self.trajectories = trajectories
        self.observations = [x.observation for x in trajectories]
        self.rewards = [x.reward for x in trajectories]
        self.actions = [x.action for x in trajectories]
        self.ratio = [x.policy_info for x in trajectories]
        self.timesteps = int(len(self.observations) / num_episodes)

    def traj2obs(self) -> np.ndarray:
        """
        Input:
            trajectories: trajectories from eval
        Output:
            np.ndarray of state observations
        """
        observations = [i.numpy()[0] for i in self.observations]
        observations = np.array(observations[:self.timesteps])
        return observations

    def traj2theta(self, obs_idx: int, acos: bool = False, asin: bool = False) -> np.ndarray:
        """
        Input:
            trajectories: trajectories from eval
            obs_idx: index of the observation corresponding to cosine or sine
        Output:
            np.ndarray of states
        """
        observations = self.traj2obs()
        theta = observations[:self.timesteps, obs_idx]

        assert not (acos and asin), '--- Error: Both acos and asin are True! ---'
        if asin:
            theta = np.arcsin(theta)
            theta = theta / np.pi * 180
        elif acos:
            theta = np.arccos(theta)
            theta = theta / np.pi * 180
        self.theta = theta
        return theta

    def traj2info(self) -> np.ndarray:
        """
        Input:
            trajectories: trajectories from eval
        Output:
            np.ndarray of info/controller ratios
        """
        try:
            ratios = [i.numpy()[0] for i in self.ratio]
        except AttributeError:
            ratios = self.ratio
        ratios = np.array(ratios[:self.timesteps])
        return ratios

    def traj2reward(self) -> float:
        """
        Input:
            trajectories: trajectories from eval
        Output:
            cumulative reward
        """
        reward = 0
        for i in self.rewards:
            reward += i.numpy()[0]
        return reward

    def _ss(self, ss=0.0):
        sample1 = self.theta[-10:]
        sample2 = self.theta[-11:-1]
        ss_reached = np.allclose(sample1, sample2, rtol=0, atol=1e-03)
        return ss_reached, self.theta[-1] - ss

    def _overshoot(self, ss_reached: bool) -> float:
        # overshoot only applicable if ss reached
        if ss_reached:
            # assume observations start at 0
            overshoot = max(np.max(self.theta), 0.0)
            if not np.isclose(overshoot, self.theta[-1], rtol=0, atol=1e-03):
                return overshoot - self.theta[-1]
            else:
                return 0.0
        else:
            return None

    def _undershoot(self, ss_reached: bool) -> float:
        # undershoot only applicable if ss reached
        if ss_reached:
            # assume observations start at 0
            undershoot = min(np.min(self.theta), 0.0)
            if not np.isclose(undershoot, self.theta[-1], rtol=0, atol=1e-03):
                return self.theta[-1] - undershoot
            else:
                return 0.0
        else:
            return None

    def _rise_time(self, percentage=0.9):
        ss_reached, ss_error = self._ss()
        if ss_reached and ss_error != 0.0:
            if ss_error > 0:
                return np.argmax(self.theta > percentage * ss_error)
            else:
                return np.argmax(self.theta < percentage * ss_error)
        else:
            return None

    def _overshoot_peak_time(self, ss_reached: bool):
        if self._overshoot(ss_reached) is None or self._overshoot(ss_reached) == 0.0:
            return None
        else:
            return self.theta.index(max(self.theta))

    def _undershoot_peak_time(self, ss_reached: bool):
        if self._overshoot(ss_reached) is None or self._undershoot(ss_reached) == 0.0:
            return None
        else:
            return self.theta.index(min(self.theta))

    def _settle_time(self, ss_target, tol=0.1) -> int:
        theta_reverse = self.theta[::-1]
        upper_stability_bound, lower_stability_bound = ss_target + tol, ss_target - tol
        theta_reverse_bool = [int(lower_stability_bound < i < upper_stability_bound) for i in theta_reverse]
        theta_reverse_bool = np.array(theta_reverse_bool)
        settling_time = np.argmin(theta_reverse_bool)
        settling_time = self.timesteps - settling_time
        return settling_time

    def _obtain_metrics(self):
        """
        Output:
            np.ndarray of the control theory metrics
        """
        # TODO: test and debug the other control metrics
        ss_reached, ss_error = self._ss()

        overshoot = self._overshoot(ss_reached)
        undershoot = self._undershoot(ss_reached)
        if overshoot is not None and undershoot is not None:
            shoot = max(abs(overshoot), abs(undershoot))
        elif overshoot is not None and undershoot is None:
            shoot = abs(overshoot)
        elif overshoot is None and undershoot is not None:
            shoot = abs(undershoot)
        else:
            shoot = None

        settle_time = self._settle_time(ss_target=ss_error, tol=np.pi / 180)

        return ss_reached, [ss_error, shoot, settle_time]


@gin.configurable
class StatePlotter(TrajectoryPath):
    # Plot the state theta and ratio for one evaluation in one env
    def __init__(self, env: PyEnvironment):
        super().__init__()
        self._init_env(env)

    def _init_env(self, env: PyEnvironment):
        self.env_name = env.unwrapped.spec.id[:-3]
        # Depending on the env, cos(theta) or state of interest is located at different obs_idx
        if self.env_name == 'Pendulum':
            self.acos = True
            self.asin = False
            self.obs_idx = 0
        elif self.env_name == 'Cartpole':
            self.acos = True
            self.asin = False
            self.obs_idx = 2
        elif self.env_name == 'Mountaincar':
            self.acos = False
            self.asin = False
            self.obs_idx = 0
        else:
            raise Exception(f'--- Error: Wrong env in {self} ---')

    def __call__(self, trajectories: List[Trajectory], num_episodes: int = 1) -> None:
        super()._helper(trajectories, num_episodes)

        thetas = self.traj2theta(obs_idx=self.obs_idx, acos=self.acos)
        ratios = self.traj2info()

        fig, ax1 = plt.subplots(figsize=(4, 3))

        # Plot theta
        ln1 = ax1.plot(np.arange(self.timesteps), thetas, color='royalblue', label='\u03B8')
        ax1.set_ylim(0, 180)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel(f'\u03B8 (\u00B0)')

        # Plot linear ratio
        ax2 = ax1.twinx()
        ln2 = ax2.plot(np.arange(self.timesteps), ratios, color='darkorange', label='Relevance r(x)')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel(f'Relevance r(x)')
        ax2.grid(False)

        # Set legend and format
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=7)
        ax1.set_title(f'State \u03B8 and Relevance r(x)')
        fig.show()


@gin.configurable
class TransientResponsePlotter(TrajectoryPath):
    # Plot the impulse and step response of all environments, with respective controllers
    def __init__(self, envs_names: List[str]):
        super().__init__()
        self.envs_names = envs_names
        self.graphs_num = len(envs_names)

    def __call__(self, all_trajectories_list: List[dict], num_episodes: int = 1) -> None:
        """
        Input:
            all_trajectories_list: a list of dict of trajectories for all envs
            num_episodes: number of evaluative episodes
        Output:
            control theory metrics
        """
        fig, axs = plt.subplots(len(all_trajectories_list), self.graphs_num, figsize=(self.graphs_num * 3, 4))

        for j, all_trajectories in enumerate(all_trajectories_list):
            for i, env_name in enumerate(self.envs_names):
                self._init_env(env_name)
                cur_axis = axs[j][i]
                self._subplot(cur_axis, all_trajectories, num_episodes, i == 0 and j == 0)

                # Set legend and format
                cur_axis.set_xlabel('Timesteps')
                if i == 0 and j == 0:
                    cur_axis.set_ylabel(f'Impulse response \u03B8 (\u00B0)')
                elif i == 0 and j == 1:
                    cur_axis.set_ylabel(f'Step response \u03B8 (\u00B0)')
                if j == 0:
                    cur_axis.set_title(f'{self.env_name}')

        fig.legend(loc=8, ncol=3)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.show()

    def _subplot(self, cur_axis, trajectories_dict, num_episodes: int, label_bool: bool):
        """
        Helper function to plot a single subplot
        :param cur_axis: the subplot object
        :param trajectories_dict: dictionary of either impulse or step trajectories
        :param num_episodes: num_episodes
        :param label_bool: only label the first column of subplots
        :return: None
        """
        for controller in trajectories_dict[self.env_name]:
            # trajectories_dict[self.env_name] contains all the trajectories for the env
            trajectories = trajectories_dict[self.env_name][controller]
            for trajectory in trajectories:
                super()._helper(trajectory, num_episodes)
                thetas = self.traj2theta(obs_idx=self.obs_idx, asin=self.asin)

                # Plot theta
                label = self.label_dict[controller] if label_bool else None
                cur_axis.plot(np.arange(self.timesteps), thetas, color=self.color_dict[controller], label=label)


@gin.configurable
class MetricsCalculator(TrajectoryPath):
    # Plot the impulse and step response of all environments, with respective controllers
    def __init__(self, envs_names: List[str]):
        super().__init__()
        self.envs_names = envs_names
        self.graphs_num = len(envs_names)

    def __call__(self, all_trajectories: dict, num_episodes: int = 1) -> None:
        """
        Input:
            all_trajectories: a dict of dict of trajectories for all envs
            num_episodes: number of evaluative episodes
        Output:
            control theory metrics
        """
        for i, env_name in enumerate(self.envs_names):
            self._init_env(env_name)
            self._controller_calc(all_trajectories, num_episodes)

    def _controller_calc(self, trajectories_dict: dict, num_episodes: int):
        """
        Helper function to get the metrics for a controller
        :param trajectories_dict: a dict of dict of trajectories for all envs
        :param num_episodes: num_episodes
        :return: None
        """
        for controller in trajectories_dict[self.env_name]:
            controller_metrics = None

            # all_trajectories[self.env_name] contains all the trajectories for the env
            trajectories = trajectories_dict[self.env_name][controller]
            for trajectory in trajectories:
                metrics = self._single_calc(trajectory, num_episodes)

                # Append the metrics to the controller_metrics AwardCurve
                if controller_metrics is None:
                    controller_metrics = metrics
                elif metrics is not None:
                    controller_metrics.append(metrics)
                else:
                    pass

            # Print out average & std of metrics
            try:
                means = controller_metrics.mean()
                stds = controller_metrics.std()
                for h, _ in enumerate(means):
                    print(f'{self.env_name} {controller} {means[h]} \u00B1 {stds[h]}')
            except AttributeError:
                print(f'{self.env_name} {controller} none stable')

    def _single_calc(self, trajectory: Trajectory, num_episodes: int):
        """
        Helper function to get the metrics from a single trajectory
        :param trajectory: trajectory
        :param num_episodes: num_episodes
        :return: metrics AwardCurve or None
        """
        super()._helper(trajectory, num_episodes)
        thetas = self.traj2theta(obs_idx=self.obs_idx, asin=self.asin)

        ss_reached, metrics = self._obtain_metrics()
        if ss_reached:
            metrics = np.array(metrics)
            x = np.zeros((len(metrics),), dtype=float)
            metrics = np.vstack((x, metrics))
            metrics = AwardCurve(metrics)
            return metrics
        else:
            return None


@gin.configurable
class RobustnessPlotter(TrajectoryPath):
    def __init__(self, envs_names: List[str]):
        super().__init__()
        self.envs_names = envs_names
        self.graphs_num = len(envs_names)
        self.label_dict['.*linear.*'] = 'Linear'

    def __call__(self, all_rewards_list: List[dict], num_episodes: int = 1) -> None:

        fig, axs = plt.subplots(len(all_rewards_list), self.graphs_num, figsize=(self.graphs_num * 3, 4))

        for j, all_rewards in enumerate(all_rewards_list):
            for i, env_name in enumerate(self.envs_names):
                self._init_env(env_name)

                cur_axis = axs[j][i]
                self._subplot(cur_axis, all_rewards[env_name], i == 0 and j == 0)

                # Set legend and format
                if j == 0:
                    cur_axis.set_xlabel('% change in mass')
                    cur_axis.set_title(f'{self.env_name}')
                elif j == 1:
                    cur_axis.set_xlabel('% change in g')
                if i == 0:
                    cur_axis.set_ylabel(f'Average cumulative rewards')

        fig.legend(loc=8, ncol=5)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.show()

    def _subplot(self, cur_axis, rewards_dict, label_bool: bool):
        """
        Helper function to plot a single subplot
        :param cur_axis: the subplot object
        :param rewards_dict: dictionary of rewards for the env
        :param label_bool: only label the first subplot
        :return: None
        """
        for policy_name in rewards_dict:
            if policy_name not in rewards_dict:
                break
            policy_rewards = rewards_dict[policy_name]
            parameter_noises = policy_rewards.x
            ymean = policy_rewards.mean()
            yerr = policy_rewards.std()

            # Plot ss_errors against parameter noise
            label = self.label_dict[policy_name] if label_bool else None
            cur_axis.errorbar(parameter_noises, ymean, yerr=yerr, color=self.color_dict[policy_name],
                              alpha=0.7, label=label)


class LearningCurvePlotter(TrajectoryPath):
    """
    This class plots learning curves for the entire training session from
    a list of np.arrays
    """

    def __init__(self):
        super().__init__()
        self.label_dict['.*linear.*'] = 'Linear/DDPG_hybrid/PILCO_hybrid'

    def __call__(self, all_curves: dict) -> None:
        """
        Input:
            all_curves: a dict of dict of all the rewards
            The content of the dictionary should be structured as such:
            dict[env_name][policy] = AwardCurves object for the env and policy
        """
        self.envs_names = all_curves.keys()
        self.graphs_num = len(self.envs_names)

        policies_list = [['ddpg_baseline', 'ddpg_hybrid'], ['pilco_baseline', 'pilco_hybrid']]
        rows = len(policies_list)

        # Plot graph
        fig, axs = plt.subplots(rows, self.graphs_num, figsize=(self.graphs_num * 3, 4))

        for i, env_name in enumerate(self.envs_names):
            for j, policies in enumerate(policies_list):
                cur_axis = axs[j][i]
                self._subplot(cur_axis, all_curves[env_name], policies, i == 0)

                # Set legend and format
                cur_axis.set_xlabel('Interaction time (s)')
                if i == 0:
                    cur_axis.set_ylabel('Normalised rewards')
                if j == 0:
                    cur_axis.set_title(env_name)

        fig.legend(loc=8, ncol=4)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    def _subplot(self, cur_axis, rewards_dict, policies: List[str], label_bool: bool):
        """
        Helper function to plot a single subplot
        :param cur_axis: the subplot object
        :param rewards_dict: dictionary of rewards for the env
        :param policies: the list of policies in the environment
        :param label_bool: only label the first column of subplots
        :return: None
        """
        for policy_name in policies:
            if policy_name not in rewards_dict:
                break

            curve = rewards_dict[policy_name]
            xs = curve.x
            curve.normalise()

            mean = curve.mean()
            std = curve.std()
            color = self.color_dict[policy_name]

            # Plot learning curve for the current env and policy
            label1 = self.label_dict[policy_name] if label_bool else None
            label2 = self.label_dict[policy_name] + ' \u00B1 std' if label_bool else None
            cur_axis.plot(xs, mean, color=color, label=label1)
            cur_axis.fill_between(xs, mean - std, mean + std, alpha=0.5, facecolor=color, label=label2)


class RegexDict(dict):

    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        for k, v in self.items():
            if re.match(k, item):
                return v
        raise KeyError


class AwardCurve(object):
    def __init__(self, data: np.ndarray):
        self.x = data[0]
        self.y = data[1:]

    def append(self, other):
        # Append other to self
        assert len(self.x) == len(other.x), '--- Error: Different timesteps ---'
        self.y = np.vstack((self.y, other.y))

    def h_append(self, other):
        # Append other to self
        self.x = np.hstack((self.x, other.x))
        self.y = np.hstack((self.y, other.y))

    def _check_dim(self):
        if self.y.ndim == 1:
            self.y = np.expand_dims(self.y, axis=0)

    def mean(self):
        self._check_dim()
        return np.mean(self.y, axis=0)

    def best(self):
        self._check_dim()
        return np.max(self.y, axis=0)

    def worst(self):
        self._check_dim()
        return np.min(self.y, axis=0)

    def std(self):
        self._check_dim()
        return np.std(self.y, axis=0)

    def normalise(self):
        self.y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
