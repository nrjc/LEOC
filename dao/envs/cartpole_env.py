"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import List

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

float_type = np.float64


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, top=False, scaling_ratio = 1.0):
        self.gravity = 9.8
        self.masscart = 0.05 * scaling_ratio
        self.masspole = 0.005 * 1./scaling_ratio
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_max = 2.5
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.top = top
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(float_type).max,
                         # self.theta_threshold_radians * 2,
                         1.,
                         1.,
                         np.finfo(float_type).max],
                        dtype=float_type)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=float_type
        )
        self.action_space = spaces.Box(
            low=-self.force_max,
            high=self.force_max,
            shape=(1,),
            dtype=float_type
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.weights = np.diag([2.0, 0.3, 2.0, 0.3])
        self.target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )

        reward = - 0.1 * (5 * (angle_normalize(theta) ** 2) + (x ** 2) + .05 * (force ** 2))
        if done:
            reward -= 100

        return self._get_obs(), reward, done, {}

    def reset(self):
        if self.top:
            self.state = np.array([0.0, 0.0, -np.pi / 180 * 1, 0.0])
        else:
            self.state = [0.0, 0.0, np.pi, 0.0]
            high = np.array([0.1, 0.1, np.pi / 180 * 10, 0.1])
            noise = self.np_random.uniform(low=-high, high=high)
            self.state = self.state + noise
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot]).reshape(-1)
        return np.array(obs, dtype=float_type)

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.__dict__[k] = self.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # M := mass of cart
        # m := mass of pole
        # l := length of pole from end to centre
        # b := coefficient of friction of cart. 'CartPoleEnv' env is frictionless
        b = 0
        M = self.masscart
        m = self.masspole
        l = self.length
        g = self.gravity
        I = 1 / 3 * m * (l ** 2)
        p = I * (M + m) + M * m * (l ** 2)

        # using x to approximate sin(x) and 1 to approximate cos(x)
        A = np.array([[0, 1, 0, 0],
                      [0, -(I + m * (l ** 2)) * b / p, (m ** 2) * g * (l ** 2) / p, 0],
                      [0, 0, 0, 1],
                      [0, (m * l * b) / p, m * g * l * (M + m) / p, 0]])

        B = np.array([[0],
                      [-(I + m * (l ** 2)) / p],
                      [0],
                      [m * l / p]])

        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        Q = np.diag([2.0, .3, 2.0, 0.3])

        return A, B, C, Q

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
