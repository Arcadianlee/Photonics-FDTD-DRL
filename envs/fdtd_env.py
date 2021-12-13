"""
custom gym environment developed for invoking FDTD simulations used in DQN.
Author: Renjie Li. October 2021 @ NOEL.
"""

from src import FdtdRl
import random
import gym
from gym import spaces, logger
from gym import utils
from gym.utils import seeding
from collections import namedtuple, deque
from itertools import count
import subprocess, time, signal
import numpy as np
import sys
import os

sys.path.append("C:\\Program Files\\Lumerical\\v202\\api\\python\\")  # Default windows lumapi path
sys.path.append(os.path.dirname(__file__))  # Current directory


class FdtdEnv(gym.Env):
    """
    Makes changes to the physical parameters of photonic crystal structures to maximize the Q factor.
    Invokes an FDTD session to take in (dx, dy, dr) and compute the resulting Q factor.

    Observations:
    Type: Box(3)
    Num     Observation                  Min                     Max
    0       net x change                -20 nm                   20 nm
    1       net y change                -20 nm                   20 nm
    2       net r change                -10 nm                   10 nm

    Actions:
    Type: Discrete(6)
    Num   Action
    0     increase x by 0.5 nm
    1     decrease x by 0.5 nm
    2     increase y by 0.5 nm
    3     decrease y by 0.5 nm
    4     increase r by 0.25 nm
    5     decrease r by 0.25 nm

    reward:
    r = 200 - (Q_target - Q)*E-7
    where Q_target = 1E+9 is the optimal Q to be achieved.

    reset:
    At the end of each episode, states are returned to zeros.

    Episode termination:
    Episode length is more than 300,
    net x change is over +- 20nm,
    net y change is over +- 20nm,
    net r change is over +- 10nm.
    Solved requirement:
    considered solved when the reward >= 150 (i.e., Q >= 0.5E9).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        # limits for net geometrical changes (states)
        self.maxDeltaX = 100
        self.maxDeltaY = 100
        self.maxDeltaR = 50

        # actions to take (i.e. alter the geometrical parameters)
        # self.delta = 0.5e-9
        # self.DR = 0.25e-9

        self.delta = 5
        self.DR = 2.5

        high = np.array(
            [
                self.maxDeltaX * 1.5,
                self.maxDeltaY * 1.5,
                self.maxDeltaR * 1.5,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # best geometrical shift values found so far
        self.x_optim = -40
        self.y_optim = 40
        self.r_optim = 25

        self.goal = 1e+7  # optimization goal
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        netDX, netDY, netDR = self.state

        if action == 0:
            netDX = netDX + self.delta

        elif action == 1:
            netDX = netDX - self.delta

        elif action == 2:
            netDY = netDY + self.delta

        elif action == 3:
            netDY = netDY - self.delta

        elif action == 4:
            netDR = netDR + self.DR

        elif action == 5:
            netDR = netDR - self.DR

        elif action == 6:
            pass

        # perform an action in fdtd and compute Q factor
        FR = FdtdRl()
        c = 1e-9  # define conversion from m to nm
        Q, _, _, _ = FR.adjustdesignparams(netDX*c, netDY*c, netDR*c)

        # update the state
        self.state = (netDX, netDY, netDR)

        done = bool(
            netDX < -self.maxDeltaX
            or netDX > self.maxDeltaX
            or netDY < -self.maxDeltaY
            or netDY > self.maxDeltaY
            or netDR < -self.maxDeltaR
            or netDR > self.maxDeltaR
        )

        # calculate the reward
        if not done:
            r = (100 - (self.goal - Q) * 1e-5)
            reward = np.float32(r)
        elif self.steps_beyond_done is None:
            # net changes out of limit, game over
            self.steps_beyond_done = 0
            r = (100 - (self.goal - Q) * 1e-5)
            reward = np.float32(r)
            print('State out of range, done! Restarting a new episode...')
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

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):

        # self.state = np.zeros((3,), dtype=np.float32)
        self.state = (self.x_optim, self.y_optim, self.r_optim)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)








