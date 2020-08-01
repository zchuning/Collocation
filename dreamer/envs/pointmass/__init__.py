from gym.envs.registration import register

#### REAL PM ENV

register(
    id='pointmass-ebm-v0',
    entry_point='ebm.envs.pointmass.pointmass_env:Pointmass',
    max_episode_steps=500,
)

register(
    id='pointmasshard-ebm-v0',
    entry_point='ebm.envs.pointmass.pointmass_hard_env:PointmassHard',
    max_episode_steps=500,
)


#### SMART PM ENV

# register(
#     id='pointmass-ebm-v0',
#     entry_point='ebm.envs.pointmass.pointmass_smart_env:Pointmass',
#     max_episode_steps=500,
# )



#### DUMMY PM ENV

# register(
#     id='pointmass-ebm-v0',
#     entry_point='ebm.envs.pointmass.pointmass_dummy_env:Pointmass',
#     max_episode_steps=500,
# )