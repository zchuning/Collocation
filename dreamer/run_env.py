# from tools import AttrDict
import numpy as np
import skvideo.io



# import blox

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
    from utils.wrappers import MultiWorld
    env = MultiWorld('push', 1)

    # from utils.wrappers import MultiTaskMetaWorld
    # env = MultiTaskMetaWorld('sawyer_SawyerReachEnvV2', 1)
    # env._env.max_path_length = np.inf
    # env.reset()

    act_space = env.action_space
    
    obs = env.reset()[0]
    import pdb; pdb.set_trace()
    
    def get_video(steps=10, a=np.zeros(2)):
        obs = env.reset()
        # goal_obs = env.render_goal() # No goal for pointmass
        
        execution = [obs['image']]
        
        for i in range(steps):
            a = env.action_space.sample()
            obs, reward, done, _ = env.step(a)
            execution.append(obs['image'])
        
        video = np.stack(execution)
        
        skvideo.io.vwrite('pointmass.gif', video, outputdict={'-r': '10'})
    
    a = np.zeros(3)  # TODO
    get_video(10, a)
    
    env.close()
    
    def async_version():
        obs = env.reset(blocking=False)()
        a = 1
        obs, _, done = env.step(a)()