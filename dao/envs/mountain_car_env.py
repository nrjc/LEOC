"""
@author: Olivier Sigaud
Modified for PILCO by Funaizhang
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
from gym.utils import seeding

float_type = np.float32


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.position_min = -2 * np.pi
        self.position_max = 1.5 * np.pi
        self.x_offset = 0.0
        self.x_scale = 1.0
        self.y_offset = 2.0
        self.y_scale = 1.0
        self.starting_position = (-np.pi - self.x_offset) / self.x_scale
        self.goal_position = (0.0 - self.x_offset) / self.x_scale

        self.gravity = 9.8
        self.masscart = 0.1
        self.force_max = 3.0
        self.tau = 0.02  # seconds between state updates

        self.low_state = np.array(
            [self.position_min, -np.finfo(np.float32).max], dtype=np.float32
        )
        self.high_state = np.array(
            [self.position_max, np.finfo(np.float32).max], dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-self.force_max,
            high=self.force_max,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.reset()

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
            position < self.position_min
            or position > self.position_max
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = np.array([self.starting_position, 0])
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

        world_width = self.position_max - self.position_min
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.position_min, self.position_max, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.position_min)*scale, ys*scale))

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
            flagx = (self.goal_position-self.position_min)*scale
            flagy1 = self._height(self.goal_position)*scale
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
            (pos-self.position_min) * scale, self._height(pos) * scale
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

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None