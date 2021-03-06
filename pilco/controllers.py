from typing import Tuple, List

import gin
import tensorflow as tf
from scipy.special import owens_t
from tensorflow_probability import distributions as tfd
import numpy as np
import gpflow
from gpflow import Parameter
from gpflow import set_trainable
from gpflow.utilities import positive
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types

from dao.controller_utils import to_distribution
from dao.envloader import TFPy2Gym
from .models import MGPR

float_type = gpflow.config.default_float()


def squash_sin(m, s, max_action=None):
    '''
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''
    k = tf.shape(m)[1]
    if max_action is None:
        max_action = tf.ones((1, k), dtype=float_type)  # squashes in [-1,1] by default
    else:
        max_action = max_action * tf.ones((1, k), dtype=float_type)

    M = max_action * tf.exp(-tf.linalg.diag_part(s) / 2) * tf.sin(m)

    lq = -(tf.linalg.diag_part(s)[:, None] + tf.linalg.diag_part(s)[None, :]) / 2
    q = tf.exp(lq)
    S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) \
        - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m)

    S = max_action * tf.transpose(max_action) * S / 2

    C = max_action * tf.linalg.diag(tf.exp(-tf.linalg.diag_part(s) / 2) * tf.cos(m))
    return M, S, tf.reshape(C, shape=[k, k])


def squash_cum_normal(m, s, max_action=None):
    '''
    Squashing function, passing the controls mean and variance through a
    cumulative normal distribution. The output is in [-max_action, max_action].
    Adapted from Rasmussen and Williams (2006) Chapter 3.9
    IN: mean (m) and variance (s) of the control input
        mean (cum_s) and variance (cum_s) of the cumulative gaussian distribution
        max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''
    if tf.shape(m)[0] != 1 and tf.shape(m)[1] != 1 or tf.shape(s)[0] != 1 and tf.shape(s)[1] != 1:
        raise Exception('Cumulative Gaussian squash function only implemented for 1 dim')

    k = tf.shape(m)[1]
    if max_action is None:
        max_action = tf.ones((1, k), dtype=float_type)  # squashes in [-1,1] by default
    else:
        max_action = max_action * tf.ones((1, k), dtype=float_type)

    m = tf.cast(m, dtype=float_type)
    s = tf.cast(s, dtype=float_type)
    # Enforce the cumulative normal distribution to be standard normal
    cum_m = tf.zeros((1, k), dtype=float_type)
    cum_s = tf.linalg.diag(tf.ones(k, dtype=float_type))
    sigma = tf.linalg.sqrtm(cum_s)
    std_normal = tfd.Normal(loc=tf.zeros((1, k), dtype=float_type), scale=sigma)

    sqrt_inv_s = tf.linalg.sqrtm(tf.linalg.inv(s + cum_s))
    z = (m - cum_m) * sqrt_inv_s
    N_z = std_normal.prob(z)
    Phi_z = std_normal.cdf(z)

    # Rasmussen and Williams (2006) Chapter 3.9 Eq. 3.82
    M = max_action * Phi_z
    # Owen, D. (1980) A table of normal integrals. Communications in Statistics: Simulation and Computation. Eq. 20,010.4
    owenT_arg1, owenT_arg2 = z, tf.linalg.sqrtm(tf.linalg.inv(1 + 2 * s))
    owenT = owens_t(owenT_arg1, owenT_arg2)
    S = Phi_z - 2 * owenT - tf.matmul(Phi_z, Phi_z)
    S = max_action * tf.transpose(max_action) * S
    # Rasmussen and Williams (2006) Chapter 3.9 Eq. 3.84
    C_intermediate = s * sqrt_inv_s * N_z + m * Phi_z
    C = C_intermediate - tf.matmul(m, Phi_z)
    C = max_action * C

    return M, S, C


@gin.configurable
class LinearController(gpflow.Module, TFPolicy):
    def __init__(self, env: TFPyEnvironment, W=None, b=None, trainable=False):
        gpflow.Module.__init__(self)
        self.state_dim = env.observation_spec().shape[0]
        self.control_dim = env.action_spec().shape[0]
        self.max_action = float(env.action_spec().maximum.max())
        self.r = 1.0
        info_spec = BoundedArraySpec((1,), float_type, minimum=0.0, maximum=1.0)
        TFPolicy.__init__(self, time_step_spec=env.time_step_spec(), action_spec=env.action_spec(), info_spec=info_spec)
        self.policy_name = 'linear'

        if W is None:
            self.W = Parameter(np.random.rand(self.control_dim, self.state_dim), dtype=float_type, trainable=trainable)
        else:
            self.W = Parameter(W, dtype=float_type, trainable=trainable)
        self.b = Parameter(np.zeros((1, self.control_dim)), dtype=float_type, trainable=trainable)

    def compute_action(self, m, s, squash=True):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M = tf.reshape(m, (-1, self.state_dim)) @ tf.transpose(self.W) + self.b  # mean output
        S = self.W @ s @ tf.transpose(self.W)  # output variance
        V = tf.transpose(self.W)  # input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        obs = time_step.observation
        M, _, _ = self.compute_action(obs, tf.zeros((self.state_dim, self.state_dim), float_type), True)

        action_distribution = tf.nest.map_structure(to_distribution, M)
        policy_state = ()  # Need to confirm policy state is not needed
        ratio = self.r
        return policy_step.PolicyStep(action_distribution, policy_state, ratio)

    def randomize(self):
        mean = 0
        sigma = 1
        self.W.assign(mean + sigma * np.random.normal(size=self.W.shape))
        self.b.assign(mean + sigma * np.random.normal(size=self.b.shape))


class FakeGPR(gpflow.Module):
    def __init__(self, data, kernel, X=None, likelihood_variance=1e-4):
        gpflow.Module.__init__(self)
        if X is None:
            self.X = Parameter(data[0], name="DataX", dtype=gpflow.default_float())
        else:
            self.X = X
        self.Y = Parameter(data[1], name="DataY", dtype=gpflow.default_float())
        self.data = [self.X, self.Y]
        self.kernel = kernel
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(likelihood_variance)
        set_trainable(self.likelihood.variance, False)


@gin.configurable
class RbfController(MGPR, TFPolicy):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''

    def __init__(self, env: TFPyEnvironment, num_basis_functions=60):
        self.state_dim = env.observation_spec().shape[0]
        self.control_dim = env.action_spec().shape[0]
        self.max_action = float(env.action_spec().maximum.max())
        self.r = 0.0
        MGPR.__init__(self,
                      [np.random.randn(num_basis_functions, self.state_dim),
                       0.1 * np.random.randn(num_basis_functions, self.control_dim)])
        info_spec = BoundedArraySpec((1,), float_type, minimum=0.0, maximum=1.0)
        TFPolicy.__init__(self, time_step_spec=env.time_step_spec(), action_spec=env.action_spec(), info_spec=info_spec)
        for model in self.models:
            model.kernel.variance.assign(1.0)
            set_trainable(model.kernel.variance, False)
        self.policy_name = 'pilco_baseline'

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):
            kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1], ], dtype=float_type))
            transformed_lengthscales = Parameter(kernel.lengthscales, transform=positive(lower=1e-3))
            kernel.lengthscales = transformed_lengthscales
            kernel.lengthscales.prior = tfd.Gamma(gpflow.utilities.to_default_float(1.1),
                                                  gpflow.utilities.to_default_float(1 / 10.0))
            if i == 0:
                self.models.append(FakeGPR((data[0], data[1][:, i:i + 1]), kernel))
            else:
                self.models.append(FakeGPR((data[0], data[1][:, i:i + 1]), kernel, self.models[-1].X))

    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        with tf.name_scope("controller") as scope:
            iK, beta = self.calculate_factorizations()
            M, S, V = self.predict_given_factorizations(m, s, 0.0 * iK, beta)
            S = S - tf.linalg.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        obs = time_step.observation
        M, _, _ = self.compute_action(obs, tf.zeros((self.state_dim, self.state_dim), float_type), True)

        action_distribution = tf.nest.map_structure(to_distribution, M)
        policy_state = ()  # Need to confirm policy state is not needed
        ratio = self.r
        return policy_step.PolicyStep(action_distribution, policy_state, ratio)

    def randomize(self):
        print("Randomising controller")
        for m in self.models:
            m.X.assign(np.random.normal(size=m.data[0].shape))
            m.Y.assign(self.max_action / 10 * np.random.normal(size=m.data[1].shape))
            mean = 1
            sigma = 0.1
            m.kernel.lengthscales.assign(mean + sigma * np.random.normal(size=m.kernel.lengthscales.shape))

    def linearize(self, loc: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """ Linearize the RBF controller about a certain point loc, and return the weight and bias for each of
        the output dimensions.

        Args:
            loc: point about which to linearize [state_dim, 1]

        Returns: Returns a list of tuples

        """
        return self.linearize_math(loc)

    def linearize_sample(self, loc: np.ndarray) -> Tuple[np.ndarray, int]:
        """ Linearize the RBF controller about a certain point loc, and return the weight and bias for each of
        the output dimensions.

        Args:
            loc: point about which to linearize [state_dim, 1]

        Returns: Returns a list of tuples

        """
        N = loc.shape[0]

        mid_p = self.compute_action(tf.reshape(tf.convert_to_tensor(loc), (1, -1)),
                                    tf.zeros([N, N], dtype=float_type),
                                    squash=True)[0].numpy()
        epsilon = 1e-6
        weight = np.zeros_like(loc)
        for i in range(len(weight)):
            loc_temp = np.copy(loc)
            small_delta = np.zeros_like(loc)
            small_delta[i] = epsilon
            loc_temp += small_delta
            weight[i] = (self.compute_action(tf.reshape(tf.convert_to_tensor(loc_temp), (1, -1)),
                                             tf.zeros([N, N], dtype=float_type),
                                             squash=True)[0] - mid_p) / epsilon
        bias = mid_p - weight @ loc

        return (weight, bias)

    def linearize_math(self, loc: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """ Linearize the RBF controller about a certain point loc, and return the weight and bias for each of
        the output dimensions.

        Args:
            loc: point about which to linearize [state_dim, 1]

        Returns: Returns a list of tuples

        """
        total_rbfs = self.num_datapoints
        returnable_obj = []
        state_dim = self.num_dims
        bias_term_collector = 0
        weight_term_collector = 0
        for m in self.models:
            centers = m.data[0][:, :].numpy()
            f_weight = m.data[1][:, 0].numpy()
            for center, weight in zip(centers, f_weight):
                lengthscale = np.diag(np.square(1 / m.kernel.lengthscales.numpy()))
                temp = (loc - center).reshape(state_dim, 1)
                exp_term = m.kernel.variance.numpy() * np.exp(
                    -0.5 * (temp.T @ lengthscale @ temp).item())  # Check that it is truly 1/var and not var
                differential_term = -exp_term * (lengthscale @ temp)
                bias_term_collector += weight * (exp_term - differential_term.T @ loc.reshape(state_dim, 1)).item()
                weight_term_collector += weight * differential_term
            returnable_obj.append((weight_term_collector, bias_term_collector))
            bias_term_collector = 0
            weight_term_collector = 0
        return returnable_obj


@gin.configurable
class HybridController(gpflow.Module, TFPolicy):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''

    def __init__(self, env: TFPyEnvironment, W=None, num_basis_functions=60):
        gpflow.Module.__init__(self)
        self.state_dim = env.observation_spec().shape[0]
        self.control_dim = env.action_spec().shape[0]
        self.max_action = float(env.action_spec().maximum.max())
        info_spec = BoundedArraySpec((1,), float_type, minimum=0.0, maximum=1.0)
        TFPolicy.__init__(self, time_step_spec=env.time_step_spec(), action_spec=env.action_spec(), info_spec=info_spec)

        self.rbf_controller = RbfController(env, num_basis_functions)
        self.linear_controller = LinearController(env, W=W, trainable=False)

        controller_location = TFPy2Gym(env).target
        self.a = Parameter(controller_location, trainable=False)
        self.S = Parameter(np.ones(self.state_dim, float_type), trainable=True, transform=positive(1e-4))
        self.r = 1.0
        self.policy_name = 'pilco_hybrid'

    def compute_ratio(self, x):
        '''
        Compute the ratio of the linear controller
        '''
        S = self.S._value()
        a = self.a._value()
        d = (x - a) @ tf.linalg.diag(S) @ tf.transpose(x - a)
        ratio = 1 / tf.pow(d + 1, 2)
        return ratio

    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        self.r = self.compute_ratio(m)
        M1, S1, V1 = self.linear_controller.compute_action(m, s, False)
        M2, S2, V2 = self.rbf_controller.compute_action(m, s, False)
        M = self.r * M1 + (1 - self.r) * M2
        S = self.r * S1 + (1 - self.r) * S2 + self.r * (M1 - M) @ tf.transpose(M1 - M) + (1 - self.r) * (
                M2 - M) @ tf.transpose(M2 - M)
        V = self.r * V1 + (1 - self.r) * V2
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        obs = time_step.observation
        M, _, _ = self.compute_action(obs, tf.zeros((self.state_dim, self.state_dim), float_type), True)

        action_distribution = tf.nest.map_structure(to_distribution, M)
        policy_state = ()  # Need to confirm policy state is not needed
        ratio = self.r
        return policy_step.PolicyStep(action_distribution, policy_state, ratio)

    def randomize(self):
        self.rbf_controller.randomize()

    def get_S(self):
        return self.S
