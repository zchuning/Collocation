import blox
import numpy as np
import skvideo.io
from utils import wrappers, tools
from utils.tools import AttrDict


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


if __name__ == "__main__":
    # from utils.wrappers import DreamerMujocoEnv
    # env = DreamerMujocoEnv(task='pm_push')
    # env = DeskEnv()
    # from utils.wrappers import KitchenEnv
    # env = KitchenEnv(task='kitchen-partial-v0')
    # env = DreamerMujocoEnv('pm_obstacle_long_1.5')
    # from utils.wrappers import MultiWorld
    # env = MultiWorld('push', 1)

    # from utils.wrappers import MultiTaskMetaWorld
    # env = MultiTaskMetaWorld('sawyer_SawyerReachEnvV2', 1)
    # env._env.max_path_length = np.inf
    # env.reset()

    task = 'sawyer_SawyerDoorCloseEnvV2'
    env = wrappers.MetaWorld(task, 1, rand_goal=False, rand_hand=True, rand_obj=True, env_rew_scale=1)
    act_space = env.action_space
    obs = env.reset()

    def get_video(steps=10, a=np.zeros(3)):
        obs = env.reset()
        execution = [obs['image']]
        for i in range(steps):
            a = env.action_space.sample()
            obs, reward, done, _ = env.step(a)
            execution.append(obs['image'])
        video = np.stack(execution)
        skvideo.io.vwrite(f'{task}_test_rhand.gif', video, outputdict={'-r': '10'})

    get_video(30)
    env.close()

    def async_version():
        obs = env.reset(blocking=False)()
        a = 1
        obs, _, done = env.step(a)()
