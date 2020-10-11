from typing import List, Union

import gin
import numpy as np
import tensorflow as tf
from gpflow import config, set_trainable
from gym import make
import pickle

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.policies.py_policy import PyPolicy

from pilco.models import PILCO
from pilco.rewards import ExponentialReward

float_type = config.default_float()


@gin.configurable
def train_pilco(env: PyEnvironment, controller: Union[PyPolicy, tf.Module], target: List[float],
                weights: List[float], m_init: List[float], S_init: List[float], initial_num_rollout: int = 3,
                training_rollout_total_num: int = 15, timesteps: int = 40, subs: int = 3,
                max_iter_policy_train: int = 50, max_training_restarts: int = 2, max_policy_restarts: int = 2) \
        -> tf.Module:

    target, weights = np.array(target), np.diag(weights)
    m_init, S_init = np.reshape(m_init, (1, -1)), np.diag(S_init)

    R = ExponentialReward(state_dim=env.observation_spec().shape[0], t=target, W=weights)
    env = env._env.gym  # Dirty hacks all around
    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(env=env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=True, verbose=False)
    for i in range(1, initial_num_rollout):
        X_, Y_, _, _ = rollout(env=env, pilco=None, timesteps=timesteps, random=True, SUBS=subs, render=True,
                               verbose=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
    pilco = PILCO((X, Y), controller=controller, horizon=timesteps, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability, we can set the likelihood variance parameters of the GP models
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    for rollouts in range(training_rollout_total_num):
        print(f'**** ITERATION no. {rollouts} ****')
        policy_restarts = 1 if rollouts > 3 else max_policy_restarts
        pilco.optimize_models(maxiter=max_iter_policy_train, restarts=max_training_restarts)
        pilco.optimize_policy(maxiter=max_iter_policy_train, restarts=policy_restarts)
        X_new, Y_new, _, _ = rollout(env, pilco, timesteps=timesteps, verbose=False, SUBS=subs)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection

        _, _, reward, intermediary_dict = pilco.predict_and_obtain_intermediates(X_new[0, None, :-1],
                                                                                 0.001 * S_init,
                                                                                 timesteps)

        # Update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
    return controller


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
    X = []
    Y = []
    x = env.reset()
    ep_return_full = 0
    ep_return_sampled = 0
    for timestep in range(timesteps):
        if render: env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, r, done, _ = env.step(u)
            ep_return_full += r
            if done: break
            if render: env.render()
        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
            print("Return so far: ", ep_return_full)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        ep_return_sampled += r
        x = x_new
        if done: break
    return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]


class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x - self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob = self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


def load_controller_from_obj(path):
    with open(path, 'rb') as f:
        con = pickle.load(f)
    return con


def save_gpflow_obj_to_path(obj, filename):
    import gpflow
    with open(filename, 'wb') as f:
        gpflow.utilities.freeze(obj)
        pickle.dump(obj, f)
