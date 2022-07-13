#Authors: Ceyao Zhang, Renjie Li, May 2022, CUHKSZ
import os
from time import time
from datetime import datetime
from tqdm import tqdm, trange

from collections import deque
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
# from torchvision.utils.data import 

import gym
# from code4.envs.fdtd_env import FdtdEnv
from environments import RewEnv, CartPoleEnv2
from models import RewNet, MLPActorCritic
from environments import FdtdEnv2

# dh - add import time
import time


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device=torch.device("cpu")):
        
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)


        def discount_cumsum(x, discount):
            """
            magic from rllab for computing discounted cumulative sums of vectors.

            input: 
                vector x, 
                [x0, 
                x1, 
                x2]

            output:
                [x0 + discount * x1 + discount^2 * x2,  
                x1 + discount * x2,
                x2]
            """
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam) ## GAE-Lambda
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an episode to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        def statistics_scalar(x):
            """
            Args:
                x: An array containing samples of the scalar to produce statistics
                    for.

                with_min_and_max (bool): If true, return min and max of x in 
                    addition to mean and std.
            """
            x = np.array(x, dtype=np.float32)
            global_sum, global_n = [np.sum(x), len(x)]
            mean = global_sum / global_n

            global_sum_sq = np.sum((x - mean)**2)
            std = np.sqrt(global_sum_sq / global_n)  # compute global std

            return mean, std

        
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std+1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, val=self.val_buf,
        ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        

        return {k: torch.as_tensor(v.copy(), dtype=torch.float32, device=self.device) for k,v in data.items()}


class RewNetBuffer:
    def __init__(self, size, device):

        self.nex_obs_buf = deque(maxlen=size)
        self.rew_buf = deque(maxlen=size)
        self.device = device

    def store(self, next_obs, rew):
        self.nex_obs_buf.append(next_obs)
        self.rew_buf.append(rew)

    def get(self):
        data = dict(next_obs=np.array(self.nex_obs_buf), rew=np.array(self.rew_buf))
        return {k: torch.as_tensor(v.copy(), dtype=torch.float32, device=self.device) for k,v in data.items()}

def main(args):

    max_steps=int(args.max_steps)
    steps_per_episode=args.steps_per_episode
    use_cuda = args.use_cuda

    gamma=args.gamma
    lam=args.lam
    clip_ratio=0.2
    target_kl=0.01

    pi_lr=3e-4
    vf_lr=1e-3
    rewNet_lr = 1e-3

    train_pi_iters=1
    train_v_iters=5

    rewNet_max_len = args.rewNet_max_len
    rewNet_batch_size = args.rewNet_batch_size
    rewNet_episodes = args.rewNet_episodes
    
    def set_seed(seed):
        # Random seed
        import random 
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)

        use_cuda = args.use_cuda
        if torch.cuda.is_available() and use_cuda:
            print("\ncuda is available! with %d gpus\n"%torch.cuda.device_count())
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda")
        else:
            print("\ncuda is not available! cpu is available!\n")
            torch.manual_seed(seed)
            device = torch.device("cpu")
        return device

    device = set_seed(args.seed)

    
    # Instantiate environment
    # env1 = gym.make("FdtdEnv")   
    # dh - change the env to CartPole
    env1 = CartPoleEnv2()
    obs_dim = env1.observation_space.shape or env1.observation_space.n
    act_dim = env1.action_space.shape or env1.action_space.n
    print(obs_dim, act_dim)

    # dh - obs_dim = (4,)
    obs_dim = obs_dim[0]

    # Create actor-critic module
    policy = MLPActorCritic(env1.observation_space, env1.action_space, hidden_sizes=[256,256])
    reward_net = RewNet(obs_dim, hidden_sizes=[32, 32])
    if torch.cuda.is_available() and use_cuda:
        policy.cuda()
        reward_net.cuda() 

    
    # Set up optimizers for policy, value function and rew Net
    pi_optimizer = Adam(policy.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(policy.v.parameters(), lr=vf_lr)
    rewNet_loss_fn = nn.MSELoss()
    rewNet_optimizer = SGD(reward_net.parameters(), lr=rewNet_lr)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_episode, gamma, lam, device)
    rewNet_buf = RewNetBuffer(rewNet_max_len, device)

    

    def update_rewNet():
        data = rewNet_buf.get()
        Xs = data['next_obs']
        ys = data['rew']
        
        dataset = TensorDataset(Xs, ys)  
        data_loader = DataLoader(dataset=dataset, batch_size=rewNet_batch_size, shuffle=True)

        reward_net.train()
        for t in range(rewNet_episodes):
            print(f"episode {t+1}\n-------------------------------")
            
            for j, (X, y) in enumerate(data_loader):
                # X, y = X.to(device), y.to(device)
                pred = reward_net(X)
                rewNet_loss = rewNet_loss_fn(pred, y)

                rewNet_optimizer.zero_grad()
                rewNet_loss.backward()
                rewNet_optimizer.step()

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = policy.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((policy.v(obs) - ret)**2).mean()

    def update_policy():
        
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

    def env_interact(obs, env, use_rewNet_flag=False):

        for t in range(steps_per_episode):

            a, v, logp = policy.step(torch.as_tensor(obs, dtype=torch.float32, device=device))
            next_o, r, d, _ = env.step(a)
            
            # save and log
            buf.store(obs, a, r, v, logp)
            if not use_rewNet_flag:
                rewNet_buf.store(next_o, r)

            obs = next_o

            episode_full = (t+1)==steps_per_episode
            if d or episode_full:
                if episode_full:
                    _, v, _ = policy.step(torch.as_tensor(obs, dtype=torch.float32, device=device))
                else:
                    v = 0.
                buf.finish_path(v)

        return obs
    
    
    
    ##########################################
    # Prepare for interaction with environment
    ##########################################

    start_time = time.time()
    o = env1.reset()

    import math
    episodes = math.ceil(max_steps/ steps_per_episode)
    for episode in trange(episodes):

        if episode < 100:
            o = env_interact(o, env1)
        else:
            # o = env1.reset()
            # o = env_interact(o, env1)
            # update_rewNet()

            if episode%10 == 0:
                o = env1.reset()
                o = env_interact(o, env1)
                update_rewNet()
            else:
                env2 = FdtdEnv2(reward_net)
                o = env2.reset()
                o = env_interact(o, env2, use_rewNet_flag=True)

        # Perform PPO update!
        update_policy()

    # dh - time() changes to time.time()
    end_time = time.time()
    print('\n Time: %.3f\n'%(end_time-start_time))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='This is PPO+RewNetSimulator hyper-parameters')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=0, choices=[0, 1], help='0 for CPU and 1 for CUDA')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor in RL')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE-Lambda')
    
    parser.add_argument('--max_steps', type=int, default=int(4e5))
    parser.add_argument('--steps_per_episode', type=int, default=200)
    
    # dh - maxlen -> max_len
    parser.add_argument('--rewNet_max_len', type=int, default=1000, help='max len for the rewNet buffer')
    parser.add_argument('--rewNet_batch_size', type=int, default=20, help='batch size for training the rewNet')
    parser.add_argument('--rewNet_epochs', type=int, default=20, help='epochs for training the rewNet')
    
    # dh - add episodes
    parser.add_argument('--rewNet_episodes', type=int, default=100, help='episodes for training the rewNet')

    args = parser.parse_args()
    print(args)

    main(args)
