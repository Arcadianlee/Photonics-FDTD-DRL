# Photonics-FDTD-DRL
Optimization and inverse design of nanoscale laser cavities using Deep Reinforcement Learning.

Abstrat:
Photonics inverse design relies on human experts to search for a design topology
that satisfies certain optical specifications with their experience and intuitions,
which is highly labor-intensive, slow, and sub-optimal. Machine learning has
emerged as a powerful tool to automate this inverse design process. However,
supervised or semi-supervised deep learning is unsuitable for this task due to: 1) a
severe shortage of available training data due to the high computational complexity
of physics-based simulations and a lack of open-source datasets; 2) the issue
of one-to-many mapping or non-unique solutions; 3) the need for a pre-trained
neural network model. Here, we propose Learning to Design Optical-Resonators
(L2DO) to leverage Reinforcement Learning (RL) that learns to autonomously
inverse design nanophotonic laser cavities without any prior knowledge while
retrieving unique design solutions. L2DO incorporates two different algorithms–
Deep Q-learning and Proximal Policy Optimization. We evaluate L2DO on two
laser cavities: a long photonic crystal (PC) nanobeam and a PC nanobeam with
an L3 cavity, both popular candidates for semiconductor lasers such as PCSELs.
Trained for less than 150 hours on limited hardware resources, L2DO has achieved
comparable or even better performance than human experts working the same
task for over a month. L2DO first learned to meet the required maxima of Q-
factors and then proceeded to optimize some additional good-to-have features (e.g.,
resonance frequency, modal volume). Compared with iterative human designs and
inverse design enabled by supervised learning, L2DO can achieve over two orders
of magnitude higher sample-efficiency without suffering from the three issues
above. This work marks the first step towards a fully automated AI framework for
photonics inverse design.

Methods:
applying deep Q learning (DQN) or Proximal Policy Optimization (PPO) to the inverse design of photonic crystal nanocavities such as the L3 cavity and the nanobeam. 

For PPO, Ray Rllib was used. For DQN, code was written from scratch.

For both cases, pytorch was used as the ML library and OpenAI gym was used for building the envs.

Here is the step-by-step instruction for how to reproduce the code in this repo:
[detailed implementation procedure of L2DO.pdf](https://github.com/Arcadianlee/Photonics-FDTD-DRL/files/9121046/detailed.implementation.procedure.of.L2DO.pdf)

To run either code in the terminal, simply type: 
python Optim_PhC_DQN.py (or Optim_PhC_PPO.py) | tee run.log

Note: you'll need your own .fsp FDTD simulation file in order for this repo to work. 

Since RL doesn't require any training data, there's no dataset included here. 

PS: for a different implementation of PPO written by one of my colleagues, see the FDTD_PPO-main folder, or visit: https://github.com/Arcadianlee/Photonics_RL



