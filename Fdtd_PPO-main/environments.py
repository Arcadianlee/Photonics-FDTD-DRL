import gym
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.registration import register

import torch
import torch.nn as nn

import numpy as np


class CartPoleEnv2(CartPoleEnv):
    def __init__(self, reward_net):
        super().__init__()
        self.reward_net = reward_net
    
    def step(self, action):
        next_o, _, done, _ = super().step(action)
        reward = self.reward_net(next_o)
    
        return next_o, reward, done, {}


class RewEnv(gym.Wrapper):
    def __init__(self, env, reward_net):
        super().__init__(env)
        self.env = env
        self.reward_net = reward_net
    
    def step(self, action):
        next_o, _, done, _ = self.env.step(action)
        reward = self.reward_net(next_o)
    
        return next_o, reward, done, {}


#register the gym env
register(
    id='Fdtd_NB-v0',
    entry_point='code4.envs:FdtdEnv',
    max_episode_steps=500,
    reward_threshold=1000.0,
)


#modify the FDTD env to incorporate NN
class FdtdEnv2(gym.Env):
    """
    Makes changes to the physical parameters of photonic crystal structures to maximize the Q factor.
    Invokes an FDTD session to take in (dx, dy, dr) and compute the resulting Q factor.
   """
    def __init__(self, NN):
        # limits for net geometrical changes (states)
        self.maxDeltaXT1 = 5
        self.maxDeltaXT2 = 5
        self.maxDeltaXT3 = 5
        self.maxDeltaXT4 = 5
        self.maxDeltaXT5 = 5
        self.maxDeltaXT6 = 5
        self.maxDeltaR = 10
        self.NN= NN
        # self.maxCav = 5

        # actions to take (i.e. alter the geometrical parameters)
        # self.delta = 0.5e-9
        # self.DR = 0.25e-

        self.delta = 0.25
        self.DR = 0.5

        high = np.array(
            [
                self.maxDeltaXT1 * 1.5,
                self.maxDeltaXT2 * 1.5,
                self.maxDeltaXT3 * 1.5,
                self.maxDeltaXT4 * 1.5,
                self.maxDeltaXT5 * 1.5,
                self.maxDeltaXT6 * 1.5,
                self.maxDeltaR * 1.5
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(14)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # best geometrical shift values found so far
        self.xt1_optim = 0
        self.xt2_optim = 0
        self.xt3_optim = 0
        self.xt4_optim = 0
        self.xt5_optim = 0
        self.xt6_optim = 0
        self.r_optim = -2.5

        self.goal = 100e+6  # optimization goal
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        netDXT1, netDXT2, netDXT3, netDXT4, netDXT5, netDXT6, netDR = self.state

        if action == 0:
            netDXT1 = netDXT1 + self.delta

        elif action == 1:
            netDXT1 = netDXT1 - self.delta

        elif action == 2:
            netDXT2 = netDXT2 + self.delta

        elif action == 3:
            netDXT2 = netDXT2 - self.delta

        elif action == 4:
            netDXT3 = netDXT3 + self.delta

        elif action == 5:
            netDXT3 = netDXT3 - self.delta

        elif action == 6:
            netDXT4 = netDXT4 + self.delta

        elif action == 7:
            netDXT4 = netDXT4 - self.delta

        elif action == 8:
            netDXT5 = netDXT5 + self.delta

        elif action == 9:
            netDXT5 = netDXT5 - self.delta

        elif action == 10:
            netDXT6 = netDXT6 + self.delta

        elif action == 11:
            netDXT6 = netDXT6 - self.delta

        elif action == 12:
            netDR = netDR + self.DR

        elif action == 13:
            netDR = netDR - self.DR

        # elif action == 14:
        #     numCav = numCav + 1
        #     if numCav > 3:
        #         numCav = 0
        #
        # elif action == 15:
        #     numCav = numCav - 1
        #     if numCav < 0:
        #         numCav = 3

        # perform an action in fdtd and compute Q factor
        #FR = FdtdRlNanobeam()
        #c = 1e-9  # define conversion from m to nm
        #Q = FR.adjustdesignparams(netDXT1*c,netDXT2*c,netDXT3*c,netDXT4*c,netDXT5*c,netDXT6*c,netDR*c)

        # update the state
        self.state = (netDXT1,netDXT2,netDXT3,netDXT4,netDXT5,netDXT6,netDR)
        self.state = np.array(self.state, dtype=np.float32)
        state = torch.from_numpy(self.state) #next obs

        #predict the reward given next state using the NN
        reward = self.NN(state).item()

        done = bool(
            netDXT1 < -self.maxDeltaXT1
            or netDXT1 > self.maxDeltaXT1
            or netDXT2 < -self.maxDeltaXT2
            or netDXT2 > self.maxDeltaXT2
            or netDXT3 < -self.maxDeltaXT3
            or netDXT3 > self.maxDeltaXT3
            or netDXT4 < -self.maxDeltaXT4
            or netDXT4 > self.maxDeltaXT4
            or netDXT5 < -self.maxDeltaXT5
            or netDXT5 > self.maxDeltaXT5
            or netDXT6 < -self.maxDeltaXT6
            or netDXT6 > self.maxDeltaXT6
            or netDR < -self.maxDeltaR
            or netDR > self.maxDeltaR
        )


        print('State: {}, reward: {}\n'.format(self.state, reward))

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        # self.state = np.zeros((4,), dtype=np.float32)
        self.state = (self.xt1_optim,self.xt2_optim,self.xt3_optim,self.xt4_optim,
                      self.xt5_optim,self.xt6_optim,self.r_optim)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)