from wrappers import DreamerMujocoEnv, MetaWorld
# from tools import AttrDict
import numpy as np
import skvideo.io
# import blox

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__
  
  
if __name__ == "__main__":
    env = MetaWorld('sawyer_SawyerPushEnvV2', 1)
    
    env._offscreen.cam.azimuth = 205
    env._offscreen.cam.elevation = -165
    env._offscreen.cam.distance = 2
    env._offscreen.cam.lookat[0] = 1.1
    env._offscreen.cam.lookat[1] = 1.1
    env._offscreen.cam.lookat[2] = -0.1

    
    act_space = env.action_space
    
    # TODO Dreamer code has some more magic with Async and other wrappers
    
    def get_video(steps=10, a=np.zeros(3)):
        obs = env.reset()
        # goal_obs = env.render_goal() # No goal for pointmass
        
        execution = [obs['image']]
        
        for i in range(steps):
            obs, reward, done, _ = env.step(a)
            execution.append(obs['image'])
        
        video = np.stack(execution)
    
        skvideo.io.vwrite('metaworld.gif', video, outputdict={'-r': '10'})

    a = np.zeros(3)  # TODO
    
    env._offscreen.cam.azimuth = 155
    env._offscreen.cam.elevation = -150
    env._offscreen.cam.distance = 0.9
    env._offscreen.cam.lookat[0] = 0.3
    env._offscreen.cam.lookat[1] = 0.55
    env._offscreen.cam.lookat[2] = -0.1
    get_video(10, a)
    

    env._offscreen.cam.azimuth = 170
    env._offscreen.cam.elevation = -165
    env._offscreen.cam.distance = 1.7
    env._offscreen.cam.lookat[0] = 1.
    env._offscreen.cam.lookat[1] = 0.5
    env._offscreen.cam.lookat[2] = -0.1
    get_video(10, a)
    
    import pdb; pdb.set_trace()
    
    env.close()
    
    
    
    def async_version():
        obs = env.reset(blocking=False)()
        a = 1
        obs, _, done = env.step(a)()