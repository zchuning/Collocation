from gym.envs.registration import register

register(
    id='push-ebm-v0',
    entry_point='ebm.envs.push.push_env:Push',
    max_episode_steps=500,
)