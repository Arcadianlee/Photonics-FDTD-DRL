# Photonics-FDTD-DRL
Optimization and inverse design of photonic crystals using Deep Reinforcement Learning.


Applying deep Q learning (DQN) or Proximal Policy Optimization (PPO) to the inverse design of photonic crystal nanocavities such as the L3 cavity and the nanobeam. 

For PPO, Ray Rllib was used. For DQN, code was written from scratch.

For both algorithms, pytorch was used. OpenAI gym is used for building the envs.

Note: you'll need your own .fsp FDTD simulation file in order for this repo to work. Since RL doesn't require any training data, there's no dataset included here. 

PS: for a different implementation of PPO written by one of my colleagues, see the FDTD_PPO-main folder, or visit: https://github.com/Arcadianlee/Photonics_RL

