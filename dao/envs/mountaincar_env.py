"""
@author: Olivier Sigaud
Modified for LEOC by Funaizhang
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
from typing import List

import numpy as np
import gym
from gym import spaces, logger
from gym.envs.classic_control import MountainCarEnv

float_type = np.float64


class ContinuousMountainCarEnv(MountainCarEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, top=False, scaling_ratio=1.0):
        self.min_position = -3.6 - np.pi / 2
        self.max_position = 2.2 - np.pi / 2
        self.x_offset = 0.0
        self.x_scale = 1.0
        self.y_offset = 1.65
        self.y_scale = 1.35
        self.starting_position = (-np.pi - self.x_offset) / self.x_scale
        self.goal_position = (0.0 - self.x_offset) / self.x_scale
        self.goal_velocity = 0.0
        self.top = top
        self.gravity = 9.8 * scaling_ratio
        self.masscart = 0.1 * scaling_ratio
        self.force_max = 3.0
        self.tau = 0.02  # seconds between state updates

        self.low_state = np.array(
            [self.min_position, -np.finfo(float_type).max], dtype=float_type
        )
        self.high_state = np.array(
            [self.max_position, np.finfo(float_type).max], dtype=float_type
        )

        self.action_space = spaces.Box(
            low=-self.force_max,
            high=self.force_max,
            shape=(1,),
            dtype=float_type
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=float_type
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.weights = np.diag([2.0, 0.4])
        self.target = np.array([self.goal_position, self.goal_velocity])

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        position, velocity = self.state
        force = action[0]

        acceleration = force / self.masscart - self.gravity * self._gradient(position) # force taken to be horizontal
        velocity += self.tau * acceleration
        position += self.tau * velocity

        self.state = np.array([position, velocity])

        # Convert a possible numpy bool to a Python bool.
        done = bool(
            position < self.min_position
            or position > self.max_position
        )

        reward = - 0.1 * (5 * (position ** 2) + (velocity ** 2) + .05 * (force ** 2))
        if done:
            reward -= 100.0

        return self._get_obs(), reward, done, {}

    def reset(self):
        if self.top:
            self.state = np.array([-np.pi * (1 / 180) / self.x_scale, 0.0])
        else:
            self.state = np.array([self.starting_position, 0])
            high = np.array([np.pi / 180 * 10 / self.x_scale, 0.1])
            noise = self.np_random.uniform(low=-high, high=high)
            self.state = self.state + noise
        self.steps_beyond_done = None
        return self._get_obs()

    def _height(self, xs):
        x = xs * self.x_scale + self.x_offset
        y = np.cos(x) * self.y_scale + self.y_offset
        return y

    def _gradient(self, xs):
        x = xs * self.x_scale + self.x_offset
        gradient = -np.sin(x) * self.y_scale
        return gradient

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        rotation = np.arctan(self._gradient(pos))
        self.cartrans.set_rotation(rotation)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_obs(self):
        position, velocity = self.state
        obs = np.array([position, velocity]).reshape(-1)
        return np.array(obs, dtype=float_type)

    def mutate_with_noise(self, noise_mag, arg_names: List[str]):
        for k in arg_names:
            self.__dict__[k] = self.__dict__[k] * (1 + np.random.uniform(-noise_mag, noise_mag))

    def control(self):
        # m := mass of car
        # b := coefficient of friction of mountain. 'MountainCarEnv' env is frictionless
        b = 0
        g = self.gravity
        m = self.masscart

        # using x to approximate sin(x)
        A = np.array([[0, 1],
                      [g, -b / m]])

        B = np.array([[0],
                      [-1 / m]])

        C = np.array([[1, 0]])

        Q = np.diag([2.0, 0.3])

        return A, B, C, Q