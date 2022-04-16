'''3rd version of NN/FDTD alternating approach, incorporating the FDTD env instead of vanilla carpole. 
   This version uses PPO, and future versions will switch to rainbow DQN. April 12nd  2022
   Authors: Renjie Li, Ceyao Zhang @CUHKSZ '''

import argparse
from email import policy
import string
import gym
import gym
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np

from typing import Optional, Union
import ray
from ray import tune

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
#from ray.rllib.evaluation.metrics import collect_metrics
#from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
#from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
#from ray.rllib.agents.ppo.ppo import PPOConfig
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
#from ray.tune.utils.placement_groups import PlacementGroupFactory

from ray.rllib.utils.sgd import do_minibatch_sgd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from collections import namedtuple, deque
from code4.envs.fdtd_env import FdtdEnv

#from code2.optim_PhC import Net
#print(torch.cuda.is_available())
#using cuda causes errors
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--num-iters", type=int, default=4)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--num-cpus-per-worker", type=int, default=5)
parser.add_argument("--framework", type=str, default='torch')
parser.add_argument("--rollfraglen", type=int, default=64)
parser.add_argument("--horizon", type=int, default=256)
parser.add_argument("--minibatch", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=20)

#parser.add_argument("--rollfraglen", type=int, default=2)

torch.set_printoptions(precision=10)

#obs space and action space dims
n_state = 7
n_actions = 14

# create a class for the NN approximating FDTD
#input: next obs, output: reward
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 120)  # just FC, no CNN
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.to(device)
        # print(x.shape)
        x = x.view(-1, n_state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



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


        print('\nState: {}, reward: {}\n'.format(self.state, reward))

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        # self.state = np.zeros((4,), dtype=np.float32)
        self.state = (self.xt1_optim,self.xt2_optim,self.xt3_optim,self.xt4_optim,
                      self.xt5_optim,self.xt6_optim,self.r_optim)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

 

#modify PPOconfig
myConfig = ppo.DEFAULT_CONFIG.copy()
args = parser.parse_args()

myConfig["framework"] = args.framework
#myConfig["rollout_fragment_length"] = args.rollfraglen
myConfig['horizon'] = args.horizon
myConfig["train_batch_size"] = args.rollfraglen
#myConfig["num_workers"] =  args.num_workers
myConfig["sgd_minibatch_size"] =  args.minibatch
myConfig["num_sgd_iter"] =  args.num_epochs
#config["batch_mode"] = "complete_episodes"


def training_workflow(config, reporter):
    '''method to function as the trainer for the RL algorithm (minicking training_iteration()
       in ppo.py) '''
    NN = Net().to(device)

    learning_rate = 0.001
    momentum = 0.9
    optimizer = optim.SGD(NN.parameters(), lr = learning_rate, momentum = momentum)

    def trainNN(data):
        '''method to optimize the NN approximating FDTD   '''
        X = data['new_obs']
        Y = data['rewards']
        X = torch.from_numpy(X).to(device)
        Y = torch.from_numpy(Y).to(device)

        print('\nupdating neural network...')

        criterion = nn.MSELoss()
        loss = criterion(torch.squeeze(NN(X)), Y) 

        # optimize the MLP model
        optimizer.zero_grad()
        loss.backward()
        for param in NN.parameters():
            # clamp grad values to between -1 and 1
            param.grad.data.clamp_(-1,1)
        optimizer.step()
        print('training loss = {}'.format(loss.item()))

    def sample_and_update(
        worker,
        num_sgd_iter: int,
        sgd_minibatch_size: int,
        standardize_fields,
    ):
        """Sample a batch and learn on it to update the policy network and KL divergence"""
        #collect samples from env and train/update policy
        T1 = SampleBatch.concat_samples([worker.sample()])
         
        info = do_minibatch_sgd(T1, worker.policy_map, worker, num_sgd_iter, sgd_minibatch_size, standardize_fields)
   
        for policy_id, policy_info in info.items():
         # Update KL loss with dynamic scaling
         # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            worker.get_policy(policy_id).update_kl(kl_divergence)

        return info, T1


    #saving sample batch data
    buffer = SampleBatch()
    totalBuffer = buffer

    env_name = 'Fdtd_NB-v0'
    register_env(env_name, lambda config: FdtdEnv())

    worker1 = RolloutWorker(
            env_creator=lambda c: FdtdEnv(), policy_spec = PPOTorchPolicy, rollout_fragment_length = args.rollfraglen,
            policy_config = myConfig, episode_horizon = args.horizon)

    register_env("nnEnv", lambda config: FdtdEnv2(NN))
    

    for i in range(config["num_iters"]):
        
        #even iteration: use FDTD
        if i % 2 == 0:
            print('\n======================\n')
            print('Using FDTD and save batch data...\n')
            
            for i in range(4): 
                # Gather a batch of samples and optimize
                std = ["advantages"]
                print('\n-----Update the policy and KL using sample batch------\n')
                info, T1 = sample_and_update(worker1, args.num_epochs, args.minibatch, std)
                buffer = SampleBatch.concat_samples([buffer, T1])   #save to buffer          
                print(buffer)

            #print(T1['new_obs'])
            #print(['actions'])
            #print(T1['rewards'])
        

            #ToDo: update the policy using policy.learn_on_batch()
            #worker1.learn_on_batch(T1)
            weights = worker1.get_weights()
            #print(weights)
            
            #T2 = SampleBatch.concat_samples([worker1.sample()])

            #buffer = SampleBatch.concat_samples([buffer, T2])   #save to buffer
            #print(buffer)

            

        #odd iteration: use NN approximating FDTD
        else:
            print('\n======================\n')
            print('........Use NN instead of FDTD........\n')  
            
            trainNN(buffer)
            buffer = SampleBatch() #clear buffer after updating

            worker2 = RolloutWorker(
                 env_creator=lambda c: FdtdEnv2(NN), policy_spec = PPOTorchPolicy, rollout_fragment_length = args.rollfraglen,
                 policy_config = myConfig, episode_horizon = args.horizon)
            worker2.set_weights(weights)
            #wt = worker2.get_weights()
            #print(wt)
            
            for i in range(4):
                # Gather a batch of samples and optimize
                std = ["advantages"]
                print('\n-----Update the policy and KL using sample batch------\n')
                info, T1 = sample_and_update(worker2, args.num_epochs, args.minibatch, std)
 
            #collect samples
            #T2 = SampleBatch.concat_samples([worker2.sample()])
            #print(T2)

            #print('\n-------Update the policy using sample batch----------\n')
            #worker2.learn_on_batch(T2)
            new_weights = worker2.get_weights()
            #print(new_weights)
            worker1.set_weights(new_weights)
    

           # reporter(**collect_metrics(remote_workers=workers))



if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus_per_worker or None)

    tune.run(
        training_workflow,
        config={
            #"batch_mode": "complete_episodes",
            "rollout_fragment_length": args.rollfraglen,
            "horizon": args.horizon,
            #"num_workers": args.num_workers,
            "num_iters": args.num_iters,
            "framework": args.framework,
            "num_gpus": 1,
            "train_batch_size": args.rollfraglen,
            #"buffer_size": 20000,
            "sgd_minibatch_size": args.minibatch,
            "num_sgd_iter": args.num_epochs,    
        },
        verbose=0,
    )

