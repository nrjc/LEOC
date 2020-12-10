import os
import pickle
import numpy as np
from typing import List

from plotting.plotter import LearningCurvePlotter, AwardCurve


def load_pickle(pickle_path: str):
    # load pickle into memory
    if os.path.isfile(pickle_path) and os.access(pickle_path, os.R_OK):
        # checks if file exists
        with open(pickle_path, 'rb') as f:
            rewards = pickle.load(f)
        return rewards
    else:
        raise Exception('--- Error: No pickle file found! ---')


def load_directory(directory: str):
    policy_curves = None

    # organise all the learning curves associated with the env and policy
    for file in os.scandir(directory):
        filename = file.name
        if filename.endswith('.pickle'):  # ignore non pickle files
            # load a single LearningCurve into memory
            path = os.path.join(directory, filename)
            data = load_pickle(path)

            # organise learning_curves for the current policy
            if policy_curves is None:
                policy_curves = data  # policy_curves is an AwardCurve object for the current policy
            else:
                policy_curves.append(data)  # policy_curves is an AwardCurve object for the current policy

    return policy_curves


def load_learning_curves(envs_names: List[str], policies: List[str]) -> dict:
    '''
    Load a dict of dict of AwardCurves from a pickle directory
    '''

    all_curves = {}  # a dict of dict containing trajectories for all envs

    # Append trajectories by env and controller
    for env_name in envs_names:
        env_curves = {}  # dict containing learning curves for the current env

        for policy in policies:
            # check the env and policy combination exists
            directory = os.path.join('pickle', env_name, policy)
            if not os.path.exists(directory):
                break

            policy_curves = load_directory(directory)

            env_curves[policy] = policy_curves
        all_curves[env_name] = env_curves
    return all_curves


if __name__ == "__main__":
    envs_names = ['Pendulum', 'Cartpole', 'Mountaincar']
    policies = ['ddpg_baseline', 'ddpg_hybrid', 'pilco_baseline', 'pilco_hybrid']

    myLearningCurveDict = load_learning_curves(envs_names, policies)
    myPlotter = LearningCurvePlotter()
    myPlotter(myLearningCurveDict)

