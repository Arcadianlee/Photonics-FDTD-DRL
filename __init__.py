# add my own enviromment to the registry, and thus make it available for gym.make()

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FdtdEnv-v0',
    entry_point='DRL.envs:FdtdEnv',
    max_episode_steps=300,
    reward_threshold=150,
)

'''
register(
    id='SoccerEmptyGoal-v0',
    entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='SoccerAgainstKeeper-v0',
    entry_point='gym.envs:SoccerAgainstKeeperEnv',
    timestep_limit=1000,
    reward_threshold=8.0,
    nondeterministic = True,
)
'''
