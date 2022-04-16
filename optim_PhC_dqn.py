"""Deep Q learning (DQN) for optimizing Nanobeam photonic crystals (using OpenAI Gym)
#Renjie Li, December 2021, NOEL CUHKSZ.
"""
import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import logging
from gym.envs.registration import register

torch.set_printoptions(precision=10)

logger = logging.getLogger(__name__)

# register the env with gym
register(
    id='Fdtd-v0',
    entry_point='envs:FdtdEnv',
    max_episode_steps=500,
    reward_threshold=75.0,
)

writer = SummaryWriter()  # log the training process

# instantiate the fdtd env
env = gym.make('Fdtd-v0').unwrapped

# if GPU is to be used
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# declare transition and experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """declare the replay buffer"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# set up the neural network
# create a class for the DQN's policy MLP
class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 50)  # just FC, no CNN
        self.fc2 = nn.Linear(50, 50)
        # self.fc3 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, num_actions)

    def forward(self, x):
        x = x.to(device)
        #         print(x.shape)
        x = x.view(-1, 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


env.reset()

# set up the training
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
UPDATE_FREQ = 4

# get number of actions from gym action space
n_actions = env.action_space.n

policy_net = Net(n_actions).to(device)  # instantiate the policy network
target_net = Net(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, momentum=0.95)  # initialize the optimizer, change learning rate?
# optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)

memory = ReplayMemory(500)  # instantiate the replay buffer

steps_done = 0  # counter for steps taken

def select_action(state):
    """selects an action accordingly to an epsilon greedy policy"""
    global steps_done
    sample = random.random()  # generate random number
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # expotentially decaying eps
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            print(policy_net(state))
            print(policy_net(state).max(1)[1])
            return policy_net(state).max(1)[1].view(1, 1)  # Pick action with the largest expected reward (argmax)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)  # pick random action

# define the optimization (RL) process, which computes V, Q and the loss
def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    print('optimizing...')

    transitions = memory.sample(BATCH_SIZE)  # sample transitions from the replay buffer
    batch = Transition(*zip(*transitions))  # transpose the batch

    # compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # state, action, and reward from replay buffer
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s')
    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # V is zero for final state
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # V' = max(Q')
    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # Q_expected = r + gamma*V'

    # cost function
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))  # L = Q.actual - Q.expected

    # optimize the MLP model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # clamp grad values to between -1 and 1
        param.grad.data.clamp_(-1,1)
    optimizer.step()
    print(loss.item())


# main training loop
num_episodes = 500
max_episode_steps = 500
tempRew = -1000
lastScore = 0
maxScore = []
for i_episode in range(num_episodes):
    # Initialize the environment and state

    print('\nStarting episode No.{}'.format(i_episode+1))

    state = env.reset()
    state = torch.from_numpy(state)
    for t in range(max_episode_steps):
        print('\nStarting time step No.{}'.format(t + 1))

        # Select and perform an action
        action = select_action(state)

        obs, score, done, _ = env.step(action.item())
        # record the highest score, corresponding to the highest Q factor
        if score > tempRew:
            tempRew = score

        # score = torch.tensor([score], device=device)

        # calculate the reward
        reward = score - lastScore
        print(score, reward, obs)

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = torch.from_numpy(obs)
        else:
            next_state = None

        if score >= env.spec.reward_threshold:
            print('\nSolved! Episode: {}, Steps: {}, Current_state: {}, Current_reward: {}\n'.format(
                i_episode, t, next_state, score))
            break

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        lastScore = score

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        # don't need to train every step
        if steps_done % UPDATE_FREQ == 0:
            optimize_model()
            writer.add_scalar('training/scores', score, steps_done)

        if done:
            break

    print('\nlargest score so far: {}'.format(tempRew))
    maxScore.append(tempRew)
    writer.add_scalar('training/max_scores', tempRew, i_episode)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        print('updating target network...')
        print(maxScore)
        target_net.load_state_dict(policy_net.state_dict())

    if score >= env.spec.reward_threshold:
        print('Solved! Episode: {}, Current_state: {}, Current_reward: {}\n'.format(
            i_episode, next_state, score))
        break



print('Training Complete')
writer.close()





